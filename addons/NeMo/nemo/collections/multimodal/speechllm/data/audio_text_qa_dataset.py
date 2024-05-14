# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import io
import json
import os
import random
from math import isclose
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import braceexpand
import numpy as np
import torch
import webdataset as wd
from omegaconf import DictConfig, ListConfig, open_dict

from nemo.collections.asr.data.audio_to_text import (
    cache_datastore_manifests,
    expand_sharded_filepaths,
    shard_manifests_if_needed,
)
from nemo.collections.asr.data.audio_to_text_dataset import ConcatDataset, convert_to_config_list, get_chain_dataset
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.asr.parts.utils.audio_utils import ChannelSelectorType
from nemo.collections.common.parts.preprocessing import collections
from nemo.collections.multimodal.speechllm.parts.utils.data_utils import (
    ceil_to_nearest,
    get_num_samples_from_files,
    maybe_cast_to_list,
)
from nemo.collections.nlp.data.language_modeling.megatron.base_dataset_utils import (
    get_datasets_weights_and_num_samples,
)
from nemo.collections.nlp.data.language_modeling.megatron.blendable_dataset import BlendableDataset
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import *
from nemo.utils import logging

try:
    from megatron.core import parallel_state, tensor_parallel

    # TODO @tmoon: Use once available in Megatron-LM
    # from megatron.core.pipeline_parallel.schedules import DataIteratorList

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False


__all__ = [
    'AudioQuestionAnswerDataset',
    'TarredAudioQuestionAnswerDataset',
    'get_tarred_aqa_dataset_from_config',
    'get_aqa_dataset_from_config',
]


def _audio_collate_fn(audio_signals, audio_lengths):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        audio_signals: List[Tensor]
        audio_lengths: List[Tensor]
    """

    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()

    audio_signals_padded = []
    for sig, sig_len in zip(audio_signals, audio_lengths):
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signals_padded.append(sig)

    if has_audio:
        audio_signals_padded = torch.stack(audio_signals_padded)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signals_padded, audio_lengths = None, None

    return audio_signals_padded, audio_lengths


def _build_loss_mask(processed_example: Dict, answer_only_loss: bool = True):
    """ Pad input_ids in batch to max batch length while building loss mask """
    # function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
    input_ids = processed_example['input_ids']
    answer_start_idx = processed_example['answer_start_idx']
    if answer_only_loss:
        loss_mask = [float(idx >= answer_start_idx) for idx in range(len(input_ids))]
    else:
        loss_mask = [1.0] * len(input_ids)

    return loss_mask


def _collate_item(item: Union[torch.Tensor, np.ndarray, List], max_length: int, pad_id: int = 0):
    # function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
    item = maybe_cast_to_list(item)
    # max_length = max([len(x) for x in item]) if item else 0
    # here [0] should be tokenizer.pad_id
    item = [x + [pad_id] * (max_length - len(x)) for x in item]
    return item


def _audio_text_collate_fn(
    batch: Dict, tokens_to_generate: int, pad_to_max_length: bool, max_seq_length: int, text_pad_id: int,
):
    sample_ids = [x["idx"] for x in batch]
    sample_ids = torch.tensor(sample_ids, dtype=torch.int32)

    audio_signal = [x["audio_signal"] for x in batch]
    audio_lengths = [x["audio_length"] for x in batch]
    audio_signal, audio_lengths = _audio_collate_fn(audio_signal, audio_lengths)

    input_ids = [item.get('masked_input_ids', item['input_ids'])[:-1] for item in batch]
    labels = [item['input_ids'][1:] for item in batch]
    contexts = [item['context_ids'] for item in batch]
    context_lengths = torch.LongTensor([item['context_length'] for item in batch])
    answers = [item['answer_ids'] for item in batch]

    loss_mask = [_build_loss_mask(item)[1:] for item in batch]

    max_length = (
        max([len(x) for x in input_ids] + [len(x) for x in answers] + [len(x) for x in contexts]) + tokens_to_generate
    )

    # increase max length to nearest multiple of 4 or 8
    if pad_to_max_length:
        max_length = max_seq_length
    else:
        max_length = min(max_seq_length, ceil_to_nearest(max_length, 8))
    assert max_length <= max_seq_length

    position_ids = [list(range(max_length)) for _ in batch]
    position_ids = torch.LongTensor(position_ids)
    input_length = torch.LongTensor([len(x) for x in input_ids])
    input_ids = torch.LongTensor(_collate_item(input_ids, max_length=max_length, pad_id=text_pad_id))
    labels = torch.LongTensor(_collate_item(labels, max_length=max_length, pad_id=text_pad_id))
    loss_mask = torch.LongTensor(_collate_item(loss_mask, max_length=max_length, pad_id=0))
    contexts = torch.LongTensor(_collate_item(contexts, max_length=max_length, pad_id=text_pad_id))
    answers = torch.LongTensor(_collate_item(answers, max_length=max_length, pad_id=text_pad_id))
    audio_ratio = [item['audio_ratio'] for item in batch if 'audio_ratio' in item]

    batch = {
        'sample_ids': sample_ids,
        'audio_signal': audio_signal,
        'audio_signal_length': audio_lengths,
        'tokens': input_ids,
        'tokens_length': input_length,
        'labels': labels,
        'loss_mask': loss_mask,
        'position_ids': position_ids,
        'contexts': contexts,
        'context_lengths': context_lengths,
        'answers': answers,
        'max_length': torch.LongTensor(max_length),
        'metadata': [x['metadata'] for x in batch],
    }
    if audio_ratio != []:
        batch['audio_ratio'] = torch.FloatTensor(audio_ratio)

    return batch


def _multi_audio_text_collate_fn(
    batch: Dict, tokens_to_generate: int, pad_to_max_length: bool, max_seq_length: int, text_pad_id: int,
):
    """Collate function for multi audio case."""
    context_start_idx = [item['context_start_idx'] for item in batch]

    audio_signals = [x["audio_signal"] for x in batch]
    audio_lengths = [x["audio_length"] for x in batch]
    num_audios = [len(x) for x in audio_signals]

    # put all audios from all samples in one batch
    audio_signals_merged = [item for audio_list in audio_signals for item in audio_list]
    audio_lengths_merged = [item for length_list in audio_lengths for item in length_list]
    audio_signals_merged, audio_lengths_merged = _audio_collate_fn(audio_signals_merged, audio_lengths_merged)

    for i in range(len(batch)):
        # create dummy audio_signal and audio_length for _audio_text_collate_fn()
        batch[i]["audio_signal"] = audio_signals[i][0]
        batch[i]["audio_length"] = audio_lengths[i][0]

    batch = _audio_text_collate_fn(batch, tokens_to_generate, pad_to_max_length, max_seq_length, text_pad_id)

    # add multi audio specific fields
    batch['context_start_idx'] = context_start_idx
    batch['num_audios'] = torch.LongTensor(num_audios)
    batch['audio_signal'] = audio_signals_merged
    batch['audio_signal_length'] = audio_lengths_merged

    return batch


class TextProcessing(object):
    """
    Text processing pipeline for AudioQuestionAnswerDataset and TarredAudioQuestionAnswerDataset.
    """

    def __init__(
        self,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        sample_alpha: Optional[float] = None,
        audio_locator: Optional[str] = None,
        input_text_mask_ratio: Optional[float] = None,
    ):
        self.input_key = input_key
        self.output_key = output_key
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_seq_length = min_seq_length
        self.seed = seed
        self.separate_prompt_and_response_with_newline = separate_prompt_and_response_with_newline
        self.answer_only_loss = answer_only_loss
        self.truncation_field = truncation_field
        self.pad_to_max_length = pad_to_max_length
        self.prompt_template = prompt_template
        self.virtual_tokens = virtual_tokens
        self.tokens_to_generate = tokens_to_generate
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_sep = add_sep
        self.end_string = end_string
        self.sample_alpha = sample_alpha
        self.audio_locator = audio_locator
        self.input_text_mask_ratio = input_text_mask_ratio

        if add_bos and hasattr(tokenizer, "bos_id") and tokenizer.bos_id > 0:
            self.bos_id = tokenizer.bos_id
        else:
            self.bos_id = None

        if add_eos and hasattr(tokenizer, "eos_id") and tokenizer.eos_id > 0:
            self.eos_id = tokenizer.eos_id
        else:
            self.eos_id = None

        if hasattr(tokenizer, "pad_id") and tokenizer.pad_id > 0:
            self.pad_id = tokenizer.pad_id
        else:
            self.pad_id = self.eos_id if self.eos_id is not None else 0

        self.sep_id = sep_id if add_sep else None

        if self.prompt_template is not None:
            # When providing things like newlines in the prompt template via the CLI, they are escaped. This line unescapes them.
            self.prompt_template = self.prompt_template.encode('utf-8').decode('unicode_escape')
        assert self.truncation_field in ["answer", "context"]

    def _random_mask_tokens(self, input_tokens, mask_ratio, mask_token, sample_tokens=None):
        output_tokens = input_tokens[:]
        mask = []
        for i, token in enumerate(input_tokens):
            prob = random.random()
            mask_or_not = prob < mask_ratio
            if mask_or_not:
                output_tokens[i] = mask_token if sample_tokens is None or i >= len(sample_tokens) else sample_tokens[i]
            mask.append(mask_or_not)
        return output_tokens, mask

    def _process_example(self, context: str, output: str):
        """
        Create an example by concatenating text and answer.
        Truncation is carried out when needed, but it is performed only on the prompt side.
        BOS, EOS, and SEP, are added if specified.

        function copied from nemo/collections/nlp/data/language_modelling/megatron/gpt_sft_dataset.py
        """
        if self.prompt_template is not None:
            assert f'{{{self.input_key}}}' in self.prompt_template
            assert f'{{{self.output_key}}}' in self.prompt_template
            # Make sure that '{output}' always occurs at the end of the prompt template string
            assert self.prompt_template.index(f'{{{self.output_key}}}') == len(self.prompt_template) - len(
                f'{{{self.output_key}}}'
            )
            # Get the context by replacing only the input
            original_context = context
            context = (
                self.prompt_template.replace(f'{{{self.input_key}}}', context)
                .replace(f'{{{self.output_key}}}', '')
                .strip(' ')
            )
            # Replace the input and output placeholders with the actual input and output
            text = self.prompt_template.replace(f'{{{self.input_key}}}', original_context).replace(
                f'{{{self.output_key}}}', output
            )

        elif self.separate_prompt_and_response_with_newline:
            text = context + '\n' + output
        else:
            text = context + ' ' + output

        if self.virtual_tokens:
            # (@adithyare) we are going to insert "pad/eos" tokens in the beginning of the text and context
            # these pad/eos tokens are placeholders for virtual tokens
            pre_pad = [self.tokenizer.eos_id] * self.virtual_tokens
        else:
            pre_pad = []
        answer_text = text[len(context) :]
        # if input_text_mask_ratio, only do it on the input but not label
        answer_ids = pre_pad + self.tokenizer.text_to_ids(
            answer_text, self.sample_alpha if self.input_text_mask_ratio is None else None
        )
        if self.end_string:
            answer_ids += self.tokenizer.text_to_ids(self.end_string)

        if self.audio_locator is None:
            # signle audio case
            context_ids = self.tokenizer.text_to_ids(context)
            context_start_idx = [0]
        else:
            # multiple audio case
            context_ids = []
            context_start_idx = []
            if self.audio_locator not in context:
                context = self.audio_locator + context
            for context_seg in context.split(self.audio_locator):
                context_start_idx.append(len(context_ids))
                context_ids.extend(self.tokenizer.text_to_ids(context_seg))
        context_ids = pre_pad + context_ids
        context_start_idx = [x + len(pre_pad) for x in context_start_idx]

        # for the long context cases, collate_fn includes self.tokens_to_generate for padding
        total_ids = len(context_ids) + max(len(answer_ids), self.tokens_to_generate)
        if self.add_bos:
            total_ids += 1
        if self.add_sep:
            total_ids += 1
        # Only training need to consider eos token
        if self.add_eos and self.tokens_to_generate == 0:
            total_ids += 1

        # If the total number of token is greater than the max, we will try to truncate the answer
        if total_ids > self.max_seq_length:
            truncation_length = total_ids - self.max_seq_length
            # TODO
            answer_ids = answer_ids[: -min(truncation_length, len(answer_ids))]
            context_ids = context_ids[: -min(truncation_length, len(context_ids))]

        input_ids = context_ids
        answer_start_idx = len(input_ids)

        # Adds bos token in the start
        if self.add_bos:
            context_ids = [self.tokenizer.bos_id] + context_ids
            input_ids = [self.tokenizer.bos_id] + input_ids
            answer_start_idx += 1

        # Adds sep token between text/prompt and answer
        if self.add_sep:
            context_ids = context_ids + [self.sep_id]
            input_ids = input_ids + [self.sep_id]
            answer_start_idx += 1

        # TODO: create a copy of answer_ids and mask on it
        if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
            if self.sample_alpha is None:
                masked_answer_ids, _ = self._random_mask_tokens(
                    answer_ids, self.input_text_mask_ratio, self.tokenizer.unk_id
                )
            else:
                sample_answer_ids = pre_pad + self.tokenizer.text_to_ids(answer_text, self.sample_alpha)
                # does not consider different length for now
                masked_answer_ids, _ = self._random_mask_tokens(
                    answer_ids, self.input_text_mask_ratio, self.tokenizer.unk_id, sample_tokens=sample_answer_ids,
                )
            masked_input_ids = input_ids + masked_answer_ids
        input_ids = input_ids + answer_ids

        # Only training need to consider eos token
        if self.add_eos and self.tokens_to_generate == 0:
            input_ids = input_ids + [self.tokenizer.eos_id]
            if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
                masked_input_ids = masked_input_ids + [self.tokenizer.eos_id]

        if len(input_ids) > self.max_seq_length:
            logging.warning(f'Input ids length {len(input_ids)} exceed max sequence length {self.max_seq_length}')
            input_ids = input_ids[: self.max_seq_length]
            if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
                masked_input_ids = masked_input_ids[: self.max_seq_length]

        processed_example = {
            'input_ids': input_ids,
            'answer_start_idx': answer_start_idx,
            'context_ids': context_ids,
            'context_length': len(context_ids),
            'answer_ids': answer_ids,
            'context_start_idx': context_start_idx,
        }

        if self.input_text_mask_ratio is not None and self.input_text_mask_ratio > 0:
            processed_example['masked_input_ids'] = masked_input_ids

        return processed_example


class AudioQuestionAnswerDataset(TextProcessing, Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        tokenizer: text tokenizer object
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        trim: whether or not to trim silence. Defaults to False
        channel_selector (int | Iterable[int] | str): select a single channel or a subset of channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
        --------- NLP SPECIFIC ARGS -------------
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        max_num_samples: Maximum number of samples to load. This can be > dataset length if you want to oversample data. If None, all samples will be loaded.
        seed: int = 1234,
        input_key: Key to use for the context in your JSONL file
        output_key: Key to use for the label in your JSONL file
        separate_prompt_and_response_with_newline: Adds a newline between prompt and response.
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: 'nemo.collections.asr.parts.perturb.AudioAugmentor' = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        channel_selector: Optional[ChannelSelectorType] = None,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: Optional[int] = None,
        max_num_samples: Optional[int] = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        index_by_file_id: bool = False,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        question_file: Optional[Union[List[str], str]] = None,
        random_context_prob: Optional[float] = None,
        random_context_num: Optional[int] = 3,
        include_voice_name: Optional[bool] = False,
        random_context_positive_percent: Optional[float] = 0.1,
        speech_prompt_in_middle: Optional[bool] = False,
        overwrite_question: Optional[bool] = False,
        sample_alpha: Optional[float] = None,
        audio_locator: Optional[str] = None,
        input_text_mask_ratio: Optional[float] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            add_sep=add_sep,
            sep_id=sep_id,
            seed=seed,
            separate_prompt_and_response_with_newline=separate_prompt_and_response_with_newline,
            answer_only_loss=answer_only_loss,
            truncation_field=truncation_field,
            pad_to_max_length=pad_to_max_length,
            prompt_template=prompt_template,
            virtual_tokens=virtual_tokens,
            tokens_to_generate=tokens_to_generate,
            input_key=input_key,
            output_key=output_key,
            end_string=end_string,
            sample_alpha=sample_alpha,
            audio_locator=audio_locator,
            input_text_mask_ratio=input_text_mask_ratio,
        )

        if isinstance(manifest_filepath, str):
            manifest_filepath = manifest_filepath.split(",")

        # If necessary, cache manifests and audio from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath, cache_audio=True)

        self.collection = collections.ALMAudioTextCollection(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
            index_by_file_id=index_by_file_id,
            max_num_samples=max_num_samples,
            question_file=question_file,
            random_context_num=random_context_num,
            include_voice_name=include_voice_name,
            random_context_positive_percent=random_context_positive_percent,
            speech_prompt_in_middle=speech_prompt_in_middle,
            overwrite_question=overwrite_question,
            random_context_prob=random_context_prob,
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.channel_selector = channel_selector

    def get_manifest_sample(self, sample_id):
        return self.collection[sample_id]

    def __getitem__(self, index):
        output = {"idx": index}
        sample = self.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        if sample.audio_file is not None:
            features = self.featurizer.process(
                sample.audio_file,
                offset=offset,
                duration=sample.duration,
                trim=self.trim,
                orig_sr=sample.orig_sr,
                channel_selector=self.channel_selector,
            )
            f, fl = features, torch.tensor(features.shape[0]).long()
            output["audio_signal"] = f
            output["audio_length"] = fl
            audio_ratio = 1
        else:
            # dummy features
            output["audio_signal"] = torch.zeros([8000])
            # accomodates normalize_batch
            output["audio_length"] = torch.tensor(8000)
            audio_ratio = 0

        text_data = self._process_example(context=sample.question, output=sample.answer)

        output.update(text_data)
        output['metadata'] = {
            'audio_filepath': sample.audio_file,
            'audio_id': sample.id,
            'offset': offset,
            'duration': sample.duration,
        }
        output['audio_ratio'] = torch.tensor(audio_ratio)
        return output

    def __len__(self):
        return len(self.collection)

    def _collate_fn(self, batch):
        return _audio_text_collate_fn(
            batch=batch,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
            text_pad_id=self.pad_id,
        )

    def collate_fn(self, batch):
        return self._collate_fn(batch)


class MultiAudioQuestionAnswerDataset(AudioQuestionAnswerDataset):
    """
    Dataset for having multi audios per sample, for example in few-shot in-context learning.
    To use this dataset, you need to specify the `audio_locator` field in the dataset config,
    and use that to specify the locations of the audio files in your manifest. In this case, 
    the `audio_filepath` field in the manifest is a list of audio filepaths, and the `duration`
    field is a list of durations, one for each audio file. The `offset` field is optional, and
    if not specified, it is assumed to be 0.0. The `offset` field is also a list of offsets if specified.

    Example manifest item for audio_locator='|audio|':
    {
    "audio_filepath": ["1.wav","2.wav","3.wav"], 
    "duration": [1.05,1.05,2.0],
    "answer": "this was her dream as nearly as she could recall it", 
    "question": "Following are examples of speech audios and their transcriptions. 
        Example 1: audio is |audio|, transcription is 'I have a dream'. 
        Example 2: audio is |audio|, transcription is ' I don't have a dream'. 
        Given the following audio |audio|, transcribe the audio into words."
    }
    """

    def __init__(
        self, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _collate_fn(self, batch):
        return _multi_audio_text_collate_fn(
            batch=batch,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
            text_pad_id=self.pad_id,
        )

    def __getitem__(self, index):
        output = {"idx": index}
        sample = self.collection[index]
        offsets = sample.offset if sample.offset else 0.0
        durations = sample.duration if sample.duration else 0.0
        num_audios = 0
        output["audio_signal"] = []
        output["audio_length"] = []
        if sample.audio_file is not None:
            audio_list = sample.audio_file
            if isinstance(sample.audio_file, str):
                audio_list = [sample.audio_file]
            if not isinstance(audio_list, list):
                raise ValueError(
                    f"The field `audio_file` must be either a str or a list of str, but got type {type(sample.audio_file)} instead"
                )

            num_audios = len(audio_list)
            if isinstance(durations, list) and len(durations) != num_audios:
                raise ValueError(
                    f"The number of durations ({len(durations)}) must match the number of audio clips ({num_audios})"
                )
            if isinstance(offsets, list) and len(offsets) != num_audios:
                raise ValueError(
                    f"The number of offsets ({len(offsets)}) must match the number of audio clips ({num_audios})"
                )

            for i, audio_file in enumerate(audio_list):
                duration = durations[i] if isinstance(durations, list) else 0
                offset = offsets[i] if isinstance(offsets, list) else 0
                features = self.featurizer.process(
                    audio_file,
                    offset=offset,
                    duration=duration,
                    trim=self.trim,
                    orig_sr=sample.orig_sr,
                    channel_selector=self.channel_selector,
                )
                f, fl = features, torch.tensor(features.shape[0]).long()
                output["audio_signal"].append(f)
                output["audio_length"].append(fl)
            audio_ratio = 1
        else:
            # dummy features
            output["audio_signal"] = [torch.zeros([8000])]
            # accomodates normalize_batch
            output["audio_length"] = [torch.tensor(8000)]
            audio_ratio = 0
            offset = 0

        text_data = self._process_example(context=sample.question, output=sample.answer)

        if isinstance(output["audio_signal"], list) and len(output["audio_signal"]) + 1 != len(
            text_data['context_start_idx']
        ):
            raise ValueError(
                f"The number of text segments ({len(text_data['context_start_idx'])}) must be one more than number of audios ({len(output['audio_signal'])})"
            )

        output.update(text_data)
        output['metadata'] = {
            'audio_filepath': sample.audio_file,
            'offset': offset,
            'duration': sample.duration,
        }
        output['audio_ratio'] = torch.tensor(audio_ratio)
        return output


class TarredAudioFilter:
    def __init__(self, collection, iterator):
        self.iterator = iterator
        self.collection = collection

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            audio_bytes, audio_filename = next(self.iterator)
            file_id, _ = os.path.splitext(os.path.basename(audio_filename))
            if file_id in self.collection.mapping:
                return audio_bytes, audio_filename


class TarredAudioLoopOffsets:
    def __init__(self, collection, iterator):
        self.iterator = iterator
        self.collection = collection
        self.current_fn = None
        self.current_bytes = None
        self.offset_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_fn is None:
            self.current_bytes, self.current_fn = next(self.iterator)
            self.offset_id = 0
        else:
            offset_list = self.collection.mapping[self.current_fn]
            if len(offset_list) == self.offset_id + 1:
                self.current_bytes, self.current_fn = next(self.iterator)
                self.offset_id = 0
            else:
                self.offset_id += 1

        return self.current_bytes, self.current_fn, self.offset_id


class TarredAudioQuestionAnswerDataset(TextProcessing, IterableDataset):
    """
    A similar Dataset to the AudioToCharDataset/AudioToBPEDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToCharDataset/AudioToBPEDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple workers the number of shards should be divisible by world_size to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToCharDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        parser (callable): A callable which is used to pre-process the text output.
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        blank_index (int): Blank character index, defaults to -1.
        unk_index (int): Unknown character index, defaults to -1.
        normalize (bool): Dataset parameter.
            Whether to use automatic text cleaning.
            It is highly recommended to manually clean text for best results.
            Defaults to True.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        bos_id (id): Dataset parameter.
            Beginning of string symbol id used for seq2seq models.
            Defaults to None.
        eos_id (id): Dataset parameter.
            End of string symbol id used for seq2seq models.
            Defaults to None.
        pad_id (id): Token used to pad when collating samples in batches.
            If this is None, pads using 0s.
            Defaults to None.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                .. warning::
                    Replicated strategy allows every node to sample the entire set of available tarfiles,
                    and therefore more than one node may sample the same tarfile, and even sample the same
                    data points! As such, there is no assured guarantee that all samples in the dataset will be
                    sampled at least once during 1 epoch. Scattered strategy, on the other hand, on specific
                    occasions (when the number of shards is not divisible with ``world_size``), will not sample
                    the entire dataset. For these reasons it is not advisable to use tarred datasets as validation
                    or test datasets.
        shard_manifests (bool): Whether or not to try / shard manifests. Defaults to False.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        --------- NLP SPECIFIC ARGS -------------
        max_seq_length (int): maximum sequence length for each dataset examples. Examples will either be truncated to fit this length or dropped if they cannot be truncated.
        min_seq_length (int): min length of each data example in the dataset. Data examples will be dropped if they do not meet the min length requirements.
        add_bos (bool): Whether to add a beginning of sentence token to each data example
        add_eos (bool): Whether to add an end of sentence token to each data example
        add_sep (bool): Whether to add a separation token to each data example (goes between prompt and answer)
        tokens_to_generate (int): (inference only) Number of tokens to generate during inference
        seed: Random seed for data shuffling.
        seed: int = 1234,
        input_key: Key to use for the context in your JSONL file
        output_key: Key to use for the label in your JSONL file
        separate_prompt_and_response_with_newline: Adds a newline between prompt and response.
        answer_only_loss: If True, will compute the loss only on the answer part of the input. If False, will compute the loss on the entire input.
        truncation_field: Field to use for truncation. (Options: "answer", "context"). Field to be used for truncation if the combined length exceeds the max sequence length.
        pad_to_max_length: Whether to pad the input to the max sequence length. If False, will pad to the max length of the current batch.
        prompt_template: Prompt template to inject via an fstring. Formatted like Q: {input}\n\nA: {output}
    """

    def __init__(
        self,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        sample_rate: int,
        int_values: bool = False,
        augmentor: Optional['nemo.collections.asr.parts.perturb.AudioAugmentor'] = None,
        shuffle_n: int = 0,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        trim: bool = False,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        question_file: Optional[Union[List[str], str]] = None,
        random_context_prob: Optional[float] = None,
        random_context_num: Optional[int] = 3,
        include_voice_name: Optional[bool] = False,
        random_context_positive_percent: Optional[float] = 0.1,
        speech_prompt_in_middle: Optional[bool] = False,
        overwrite_question: Optional[bool] = False,
        sample_alpha: Optional[float] = None,
        input_text_mask_ratio: Optional[float] = None,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            add_sep=add_sep,
            sep_id=sep_id,
            seed=seed,
            separate_prompt_and_response_with_newline=separate_prompt_and_response_with_newline,
            answer_only_loss=answer_only_loss,
            truncation_field=truncation_field,
            pad_to_max_length=pad_to_max_length,
            prompt_template=prompt_template,
            virtual_tokens=virtual_tokens,
            tokens_to_generate=tokens_to_generate,
            input_key=input_key,
            output_key=output_key,
            end_string=end_string,
            sample_alpha=sample_alpha,
            input_text_mask_ratio=input_text_mask_ratio,
        )
        self.is_megatron_iterable = True
        self.shard_manifests = shard_manifests

        # Shard manifests if necessary and possible and then expand the paths
        manifest_filepath = shard_manifests_if_needed(
            shard_manifests=shard_manifests,
            shard_strategy=shard_strategy,
            manifest_filepaths=manifest_filepath,
            world_size=world_size,
            global_rank=global_rank,
        )

        # If necessary, cache manifests from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath)

        self.collection = collections.ALMAudioTextCollection(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,
            question_file=question_file,
            random_context_prob=random_context_prob,
            random_context_num=random_context_num,
            include_voice_name=include_voice_name,
            random_context_positive_percent=random_context_positive_percent,
            speech_prompt_in_middle=speech_prompt_in_middle,
            overwrite_question=overwrite_question,
        )

        self.len = self._compute_len()

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        assert audio_tar_filepaths
        logging.info(f"WebDataset is created for {manifest_filepath} {audio_tar_filepaths}")
        audio_tar_filepaths = expand_sharded_filepaths(
            sharded_filepaths=audio_tar_filepaths,
            shard_strategy=shard_strategy,
            world_size=world_size,
            global_rank=global_rank,
        )

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio='wav;ogg;flac', key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .pipe(self._loop_offsets)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRAudioText already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """
        return TarredAudioFilter(self.collection, iterator)

    def _loop_offsets(self, iterator):
        """This function is used to iterate through utterances with different offsets for each file.
        """
        return TarredAudioLoopOffsets(self.collection, iterator)

    def _collate_fn(self, batch):
        return _audio_text_collate_fn(
            batch=batch,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
            text_pad_id=self.pad_id,
        )

    def collate_fn(self, batch):
        return self._collate_fn(batch)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename, offset_id = tup

        assert audio_filename is not None
        # Grab manifest entry from self.manifest_preprocessor.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))
        manifest_idx = self.collection.mapping[file_id][offset_id]
        manifest_entry = self.collection[manifest_idx]

        # init output dict
        output = {"idx": manifest_idx}

        offset = manifest_entry.offset
        if offset is None:
            offset = 0
        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream,
            offset=offset,
            duration=manifest_entry.duration,
            trim=self.trim,
            orig_sr=manifest_entry.orig_sr,
        )
        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()
        output["audio_signal"] = f
        output["audio_length"] = fl

        # Text features
        text_data = self._process_example(context=manifest_entry.question, output=manifest_entry.answer)

        output.update(text_data)

        output['metadata'] = {
            'audio_filepath': audio_filename,
            'offset': offset,
            'duration': manifest_entry.duration,
        }
        output['audio_ratio'] = torch.tensor(1)
        return output

    def get_manifest_sample(self, sample_id):
        return self.collection[sample_id]

    def __iter__(self):
        return self._dataset.__iter__()

    def _compute_len(self):
        # TODO: need to figure out why here needs to be divided by world_size, while in ASR we don't need to.
        if self.shard_manifests and torch.distributed.is_available() and torch.distributed.is_initialized():
            my_len = torch.tensor(len(self.collection), dtype=torch.int32).cuda()
            torch.distributed.all_reduce(my_len)
            my_len = my_len.int() // parallel_state.get_data_parallel_world_size()
            logging.info(f'Sharded manifests: Total length: {my_len}')
        else:
            my_len = len(self.collection) // parallel_state.get_data_parallel_world_size()

        return my_len

    def __len__(self):
        return self.len


class IterableTextDataset(TextProcessing, IterableDataset):
    def __getitem__(self, index):
        output = {"idx": index}
        sample = self.collection[index]
        # handle pure text case in the joint training
        assert sample.audio_file is None
        offset = sample.offset

        if offset is None:
            offset = 0

        # dummy features
        output["audio_signal"] = torch.zeros([8000])
        # accomodates normalize_batch
        output["audio_length"] = torch.tensor(8000)

        text_data = self._process_example(context=sample.question, output=sample.answer)

        output.update(text_data)
        output['metadata'] = {
            'audio_filepath': sample.audio_file,
            'offset': offset,
            'duration': sample.duration,
        }
        output['audio_ratio'] = torch.tensor(0)
        return output

    def _collate_fn(self, batch):
        return _audio_text_collate_fn(
            batch=batch,
            tokens_to_generate=self.tokens_to_generate,
            pad_to_max_length=self.pad_to_max_length,
            max_seq_length=self.max_seq_length,
            text_pad_id=self.pad_id,
        )

    def __iter__(self):
        for _ in range(self.__len__()):
            yield self.__getitem__(np.random.randint(0, len(self.collection)))

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: 'nemo.collections.common.tokenizers.TokenizerSpec',
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        shard_strategy: str = "scatter",
        shard_manifests: bool = False,
        global_rank: int = 0,
        world_size: int = 0,
        max_seq_length: int = 1024,
        min_seq_length: int = 1,
        add_bos: bool = False,
        add_eos: bool = True,
        add_sep: bool = False,
        sep_id: int = None,
        seed: int = 1234,
        separate_prompt_and_response_with_newline: bool = False,
        answer_only_loss: bool = True,
        truncation_field: str = "answer",
        pad_to_max_length: bool = False,  # (@adithyare) allows for much faster training especially in PEFT settings.
        prompt_template: str = None,
        virtual_tokens: int = 0,
        tokens_to_generate: int = 0,
        input_key: str = 'input',
        output_key: str = 'output',
        end_string: Optional[str] = None,
        question_file: Optional[str] = None,
        random_context_prob: Optional[float] = None,
        random_context_num: Optional[int] = 3,
        include_voice_name: Optional[bool] = False,
        random_context_positive_percent: Optional[float] = 0.1,
        speech_prompt_in_middle: Optional[bool] = False,
        overwrite_question: Optional[bool] = False,
        sample_alpha: Optional[float] = None,
        input_text_mask_ratio: Optional[float] = None,
        max_iter_num: Optional[int] = 1e9,
    ):
        super().__init__(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            min_seq_length=min_seq_length,
            add_bos=add_bos,
            add_eos=add_eos,
            add_sep=add_sep,
            sep_id=sep_id,
            seed=seed,
            separate_prompt_and_response_with_newline=separate_prompt_and_response_with_newline,
            answer_only_loss=answer_only_loss,
            truncation_field=truncation_field,
            pad_to_max_length=pad_to_max_length,
            prompt_template=prompt_template,
            virtual_tokens=virtual_tokens,
            tokens_to_generate=tokens_to_generate,
            input_key=input_key,
            output_key=output_key,
            end_string=end_string,
            sample_alpha=sample_alpha,
            input_text_mask_ratio=input_text_mask_ratio,
        )

        self.max_iter_num = max_iter_num
        self.shard_manifests = shard_manifests

        # Shard manifests if necessary and possible and then expand the paths
        manifest_filepath = shard_manifests_if_needed(
            shard_manifests=shard_manifests,
            shard_strategy=shard_strategy,
            manifest_filepaths=manifest_filepath,
            world_size=world_size,
            global_rank=global_rank,
        )

        # If necessary, cache manifests from object store
        cache_datastore_manifests(manifest_filepaths=manifest_filepath)

        self.collection = collections.ALMAudioTextCollection(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,
            question_file=question_file,
            random_context_prob=random_context_prob,
            random_context_num=random_context_num,
            include_voice_name=include_voice_name,
            random_context_positive_percent=random_context_positive_percent,
            speech_prompt_in_middle=speech_prompt_in_middle,
            overwrite_question=overwrite_question,
        )

        self.len = self._compute_len()

        # handle the text-only case
        logging.info(f"Non-WebDataset is created for {manifest_filepath}")

    def _compute_len(self):
        # TODO: need to figure out why here needs to be divided by world_size, while in ASR we don't need to.
        if self.shard_manifests and torch.distributed.is_available() and torch.distributed.is_initialized():
            my_len = torch.tensor(len(self.collection), dtype=torch.int32).cuda()
            torch.distributed.all_reduce(my_len)
            my_len = my_len.int() // parallel_state.get_data_parallel_world_size()
            logging.info(f'Sharded manifests: Total length: {my_len}')
        else:
            my_len = len(self.collection) // parallel_state.get_data_parallel_world_size()

        return my_len

    def __len__(self):
        return min(self.max_iter_num // parallel_state.get_data_parallel_world_size(), self.len)


def get_tarred_aqa_dataset(
    config,
    tokenizer,
    augmentor,
    global_rank=0,
    world_size=1,
    shuffle_n=0,
    sep_id=None,
    answer_only_loss=True,
    virtual_tokens=0,
):
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    tarred_audio_filepaths = convert_to_config_list(tarred_audio_filepaths)
    manifest_filepaths = convert_to_config_list(manifest_filepaths)

    bucketing_weights = config.get('bucketing_weights', None)  # For upsampling buckets
    if bucketing_weights:
        for idx, weight in enumerate(bucketing_weights):
            if not isinstance(weight, int) or weight <= 0:
                raise ValueError(f"bucket weights must be positive integers")

    if len(manifest_filepaths) != len(tarred_audio_filepaths):
        raise ValueError(
            f"manifest_filepaths (length={len(manifest_filepaths)}) and tarred_audio_filepaths (length={len(tarred_audio_filepaths)}) need to have the same number of buckets."
        )

    if 'labels' not in config:
        logging.warning(f"dataset does not have explicitly defined labels")

    if 'max_utts' in config:
        raise ValueError('"max_utts" parameter is not supported for tarred datasets')

    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        if len(tarred_audio_filepath) == 1:
            tarred_audio_filepath = tarred_audio_filepath[0]
        if len(manifest_filepath) == 1:
            manifest_filepath = manifest_filepath[0]

        question_file_set = config.get('question_file_set', None)
        if question_file_set is not None:
            assert len(question_file_set) == len(tarred_audio_filepaths)
            question_file = question_file_set[dataset_idx]
        else:
            question_file = None
        if tarred_audio_filepath == []:
            dataset = IterableTextDataset(
                manifest_filepath=manifest_filepath,
                tokenizer=tokenizer,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                shard_manifests=config.get('shard_manifests', False),
                global_rank=global_rank,
                world_size=world_size,
                max_seq_length=config.max_seq_length,
                min_seq_length=config.min_seq_length,
                add_bos=config.get('add_bos', False),
                add_eos=config.get('add_eos', True),
                add_sep=config.get('add_sep', False),
                sep_id=sep_id,
                separate_prompt_and_response_with_newline=config.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=answer_only_loss,
                truncation_field=config.get('truncation_field', 'context'),
                pad_to_max_length=False,
                prompt_template=config.get('prompt_template', None),
                virtual_tokens=virtual_tokens,
                tokens_to_generate=config.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                input_key=config.get('input_key', 'input'),
                output_key=config.get('output_key', 'output'),
                end_string=config.get('end_string', None),
                sample_alpha=config.get('sample_alpha', None),
                input_text_mask_ratio=config.get('input_text_mask_ratio', None),
                question_file=question_file,
                random_context_num=config.get('random_context_num', 3),
                include_voice_name=config.get('include_voice_name', False),
                random_context_positive_percent=config.get('random_context_positive_percent', 0.1),
                speech_prompt_in_middle=config.get('speech_prompt_in_middle', False),
                overwrite_question=config.get('overwrite_question', False),
                random_context_prob=config.get('random_context_prob', None),
                max_iter_num=config.get('max_iter_num', 1e9),
            )
        else:
            dataset = TarredAudioQuestionAnswerDataset(
                audio_tar_filepaths=tarred_audio_filepath,
                manifest_filepath=manifest_filepath,
                tokenizer=tokenizer,
                sample_rate=config['sample_rate'],
                int_values=config.get('int_values', False),
                augmentor=augmentor,
                shuffle_n=shuffle_n,
                max_duration=config.get('max_duration', None),
                min_duration=config.get('min_duration', None),
                trim=config.get('trim_silence', False),
                shard_strategy=config.get('tarred_shard_strategy', 'scatter'),
                shard_manifests=config.get('shard_manifests', False),
                global_rank=global_rank,
                world_size=world_size,
                max_seq_length=config.max_seq_length,
                min_seq_length=config.min_seq_length,
                add_bos=config.get('add_bos', False),
                add_eos=config.get('add_eos', True),
                add_sep=config.get('add_sep', False),
                sep_id=sep_id,
                separate_prompt_and_response_with_newline=config.get(
                    'separate_prompt_and_response_with_newline', True
                ),
                answer_only_loss=answer_only_loss,
                truncation_field=config.get('truncation_field', 'context'),
                pad_to_max_length=False,
                prompt_template=config.get('prompt_template', None),
                virtual_tokens=virtual_tokens,
                tokens_to_generate=config.get(
                    'tokens_to_generate', 0
                ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
                input_key=config.get('input_key', 'input'),
                output_key=config.get('output_key', 'output'),
                end_string=config.get('end_string', None),
                sample_alpha=config.get('sample_alpha', None),
                input_text_mask_ratio=config.get('input_text_mask_ratio', None),
                question_file=question_file,
                random_context_num=config.get('random_context_num', 3),
                include_voice_name=config.get('include_voice_name', False),
                random_context_positive_percent=config.get('random_context_positive_percent', 0.1),
                speech_prompt_in_middle=config.get('speech_prompt_in_middle', False),
                overwrite_question=config.get('overwrite_question', False),
                random_context_prob=config.get('random_context_prob', None),
            )

        if bucketing_weights:
            [datasets.append(dataset) for _ in range(bucketing_weights[dataset_idx])]
        else:
            datasets.append(dataset)

    with open_dict(config):  # patch for bucketing tarred datasets
        config['batch_size'] = config.get("micro_batch_size", 1)
    return get_chain_dataset(datasets=datasets, ds_config=config, rank=global_rank)


def get_concat_tarred_aqa_dataset(
    config,
    tokenizer,
    augmentor,
    global_rank=0,
    world_size=1,
    shuffle_n=0,
    sep_id=None,
    answer_only_loss=True,
    virtual_tokens=0,
):
    tarred_audio_filepaths = config['tarred_audio_filepaths']
    manifest_filepaths = config['manifest_filepath']
    datasets = []
    for dataset_idx, (tarred_audio_filepath, manifest_filepath) in enumerate(
        zip(tarred_audio_filepaths, manifest_filepaths)
    ):
        conf = copy.deepcopy(config)
        conf['manifest_filepath'] = manifest_filepath
        conf['tarred_audio_filepaths'] = tarred_audio_filepath
        question_file_set = config.get('question_file_set', None)
        conf['question_file_set'] = [question_file_set[dataset_idx]]
        dataset = get_tarred_aqa_dataset(
            config=conf,
            tokenizer=tokenizer,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            augmentor=augmentor,
            sep_id=sep_id,
            answer_only_loss=answer_only_loss,
            virtual_tokens=virtual_tokens,
        )
        datasets.append(dataset)

    concat_sampling_probabilities = config.get('concat_sampling_probabilities', None)
    if not isinstance(concat_sampling_probabilities, ListConfig) or len(concat_sampling_probabilities) != len(
        datasets
    ):
        concat_sampling_probabilities = [1.0 / len(datasets)] * len(datasets)

    dataset = ConcatDataset(
        datasets,
        sampling_technique=config.get('concat_sampling_technique', 'temperature'),
        sampling_temperature=config.get('concat_sampling_temperature', 5),
        sampling_scale=config.get('concat_sampling_scale', 1),
        sampling_probabilities=config.get('concat_sampling_probabilities', None),
        shuffle=config.get('concat_shuffle', True),
        seed=config.get('concat_sampling_seed', None),
        global_rank=global_rank,
        world_size=world_size,
    )
    return dataset


def get_tarred_aqa_dataset_from_config(
    config: DictConfig,
    tokenizer,
    augmentor,
    global_rank: int = 0,
    world_size: int = 1,
    sep_id: Optional[int] = None,
    answer_only_loss: bool = True,
    virtual_tokens: int = 0,
):
    is_concat = config.get('is_concat', False)
    if is_concat:
        if 'concat_sampling_technique' in config and config['concat_sampling_technique'] is None:
            logging.warning(
                f"Concat dataset requires `concat_sampling_technique` but it was not provided. Config: {config}"
            )
            return None
        if 'concat_sampling_technique' in config and config['concat_sampling_technique'] == 'random':
            if not 'concat_sampling_probabilities' in config:
                logging.warning(f"Concat dataset requires `concat_sampling_probabilities` list. Config: {config}")
                return None
            else:
                if not isclose(sum(config['concat_sampling_probabilities']), 1, abs_tol=1e-6):
                    logging.warning(f"`concat_sampling_probabilities` need to sum to 1. Config: {config}")
                    concat_sampling_probabilities = config['concat_sampling_probabilities']
                    concat_sampling_probabilities = [
                        float(x) / sum(concat_sampling_probabilities) for x in concat_sampling_probabilities
                    ]
                    logging.info(f"Normalized concat dataset sampling probabilities: {concat_sampling_probabilities}")
                    config['concat_sampling_probabilities'] = concat_sampling_probabilities

    data_parallel_size = parallel_state.get_data_parallel_world_size()
    num_micro_batches = config.global_batch_size // (config.micro_batch_size * data_parallel_size)
    global_batch_size_on_this_data_parallel_rank = num_micro_batches * config.micro_batch_size
    shuffle = config['shuffle']
    shuffle_n = config.get('shuffle_n', 4 * global_batch_size_on_this_data_parallel_rank) if shuffle else 0
    if is_concat:
        dataset = get_concat_tarred_aqa_dataset(
            config=config,
            tokenizer=tokenizer,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            sep_id=sep_id,
            answer_only_loss=answer_only_loss,
            virtual_tokens=virtual_tokens,
        )
    else:
        dataset = get_tarred_aqa_dataset(
            config=config,
            tokenizer=tokenizer,
            augmentor=augmentor,
            shuffle_n=shuffle_n,
            global_rank=global_rank,
            world_size=world_size,
            sep_id=sep_id,
            answer_only_loss=answer_only_loss,
            virtual_tokens=virtual_tokens,
        )
    return dataset


def get_aqa_dataset_from_config(
    manifest_filepath: str,
    config: DictConfig,
    tokenizer,
    augmentor,
    is_train,
    sep_id: Optional[int] = None,
    answer_only_loss: bool = True,
    virtual_tokens: int = 0,
):
    if isinstance(config.manifest_filepath, str):
        manifest_filepath = config.manifest_filepath.split(',')
    else:
        manifest_filepath = config.manifest_filepath

    data_cls = MultiAudioQuestionAnswerDataset if config.get('audio_locator', None) else AudioQuestionAnswerDataset
    datasets = []
    if is_train:
        # Construct the data prefix list for `get_datasets_weights_and_num_samples()`
        # that is of the format [weight1,file_name1,weight2,file_name2,...]
        concat_sampling_probabilities = config.get('concat_sampling_probabilities', None)
        if concat_sampling_probabilities is None:
            concat_sampling_probabilities = [1.0 / len(manifest_filepath)] * len(manifest_filepath)
        elif len(config.get('concat_sampling_probabilities', None)) != len(manifest_filepath):
            raise ValueError(
                (
                    f"concat_sampling_probabilities must be of the same size as manifest_filepath.",
                    f"Provided size {len(config.concat_sampling_probabilities)}, number of datasets {len(manifest_filepath)}",
                )
            )
        data_prefix = []
        for weight, prefix in zip(concat_sampling_probabilities, manifest_filepath):
            data_prefix.append(weight)
            data_prefix.append(prefix)

        num_samples_per_dataset = get_num_samples_from_files(manifest_filepath)
        num_train_samples = [len(manifest_filepath) * max(num_samples_per_dataset)]
        _, _, num_train_samples_per_dataset = get_datasets_weights_and_num_samples(data_prefix, num_train_samples)
        num_train_samples_after_blend = sum([x[0] for x in num_train_samples_per_dataset])
    else:
        num_train_samples_per_dataset = [[None]] * len(manifest_filepath)

    for dataset_idx, (file_path, num_samples) in enumerate(zip(manifest_filepath, num_train_samples_per_dataset)):
        question_file_set = config.get('question_file_set', None)
        if question_file_set is not None:
            assert len(question_file_set) == len(manifest_filepath)
            question_file = question_file_set[dataset_idx]
        else:
            question_file = None
        dataset = data_cls(
            manifest_filepath=file_path,
            tokenizer=tokenizer,
            sample_rate=config.sample_rate,
            int_values=config.get('int_values', False),
            augmentor=augmentor,
            max_duration=getattr(config, 'max_duration', None),
            min_duration=getattr(config, 'min_duration', None),
            max_utts=getattr(config, 'max_utts', -1),
            trim=getattr(config, 'trim_silence', False),
            channel_selector=getattr(config, 'channel_selector', None),
            max_seq_length=config.max_seq_length,
            min_seq_length=config.min_seq_length,
            add_bos=config.get('add_bos', False),
            add_eos=config.get('add_eos', True),
            add_sep=config.get('add_sep', False),
            sep_id=sep_id,
            max_num_samples=num_samples[0],
            seed=config.get('seed', 1234),
            separate_prompt_and_response_with_newline=config.get('separate_prompt_and_response_with_newline', True),
            answer_only_loss=answer_only_loss,
            truncation_field=config.get('truncation_field', 'context'),
            pad_to_max_length=config.get('pad_to_max_length', False),
            prompt_template=config.get('prompt_template', None),
            virtual_tokens=virtual_tokens,
            tokens_to_generate=config.get(
                'tokens_to_generate', 0
            ),  # used at inference time to allocate tensor positions for tokens that will be generated by inf procedure.
            input_key=config.get('input_key', 'input'),
            output_key=config.get('output_key', 'output'),
            end_string=config.get('end_string', None),
            sample_alpha=config.get('sample_alpha', None),
            input_text_mask_ratio=config.get('input_text_mask_ratio', None),
            random_context_prob=config.get('random_context_prob', None),
            random_context_num=config.get('random_context_num', 3),
            include_voice_name=config.get('include_voice_name', False),
            random_context_positive_percent=config.get('random_context_positive_percent', 0.1),
            speech_prompt_in_middle=config.get('speech_prompt_in_middle', False),
            overwrite_question=config.get('overwrite_question', False),
            question_file=question_file,
            audio_locator=config.get('audio_locator', None),
        )
        datasets.append(dataset)

    if is_train:
        dataset = BlendableDataset(
            datasets=datasets, weights=concat_sampling_probabilities, size=num_train_samples_after_blend
        )
        return dataset
    else:
        return datasets
