import argparse
import json
import os

LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "it": "italian",
    "pl": "polish",
    "pt": "portuguese",
    "nl": "dutch",
    "ca": "catalan",
    "ar": "arabic",
    "cy": "welsh",
    "et": "estonian",
    "id": "indonesian",
    "lv": "latvian",
    "sl": "slovenian",
    "ta": "tamil",
    "zh": "chinese",
    "fa": "persian",
    "ja": "japanese",
    "mn": "mongolian",
    "sv": "swedish",
    "tr": "turkish",
}

def read_file(filepath):
    ret = []
    with open(filepath, 'r') as f:
        for line in f:
            id_, content = line.strip().split(' ', 1)
            ret.append((id_, content))
    return ret

def format_data(audio_id, audio_path, task, model, dataset, text, src_lang, tgt_lang, instruction='', instruction_modifier='', separators=[]):
    output_dict = {
        "audio_id": audio_id,
        "audio_path": audio_path,
        "dataset": dataset.lower(),
        "task": task,
    }
    if model == 'ltu_as':
        if task == 'asr':
            assert instruction == '', "A fixed instruction should be used in this task."
            output_dict.update(
                {
                    "instruction": "Closed-ended question: Can you identify the spoken text?",
                    "input": "", # This should be the Whisper ASR result.
                    "output": f"Spoken text: {text}",
                }
            )
        elif task in ['slu-sa', 'slu-summ', 'slu-dac', 'dynamic_superb-classification']:
            assert instruction != '', "Please specify the instruction to be used in this task."
            output_dict.update(
                {
                    "instruction": instruction,
                    "input": "", # This should be the Whisper ASR result.
                    "output": text,
                }
            )
        elif task in ['slu-sqa']:
            assert separators != [], "Please specify the separator to be used in this task."
            output_dict.update(
                {
                    "instruction": text.split(separators[0])[0].strip(),
                    "input": "", # This should be the Whisper ASR result.
                    "output": text.split(separators[1])[1].strip(),
                }
            )
    elif model == 'salmonn':
        if task == 'asr':
            assert instruction == '', "A fixed instruction should be used in this task."
            output_dict.update(
                {
                    "instruction": f"Recognize the speech and give me the transcription{instruction_modifier}.",
                    "output": text,
                } 
            )
        elif task == 'st':
            assert instruction == '', "A fixed instruction should be used in this task."
            output_dict.update(
                {
                    "instruction": f"Listen to the speech{instruction_modifier} and translate it into {LANGUAGES[tgt_lang].capitalize()}.",
                    "output": text[1],
                    "source_text": text[0],
                    "target_text": text[1],
                } 
            )
            if src_lang == 'en' and tgt_lang == 'ja':
                # According to the author of SALMONN, this leads to better en-ja translation performance
                if instruction_modifier:
                    output_dict["instruction"] = "英語のスピーチを聞いて日本語に訳す"
                else:
                    output_dict["instruction"] = "スピーチを聞き、日本語に訳す"
        elif task in ['slu-sa', 'slu-summ', 'slu-dac', 'dynamic_superb-classification']:
            assert instruction != '', "Please specify the instruction to be used in this task."
            output_dict.update(
                {
                    "instruction": instruction,
                    "output": text,
                }
            )
        elif task in ['slu-sqa']:
            assert separators != [], "Please specify the separator to be used in this task."
            output_dict.update(
                {
                    "instruction": text.split(separators[0])[0].strip(),
                    "output": text.split(separators[1])[1].strip(),
                }
            )
    elif model == 'qwen_audio':
        if task == 'asr':
            assert instruction == '', "A fixed instruction should be used in this task."
            output_dict.update(
                {
                    "instruction": f"<|startoftranscript|><|{src_lang}|><|transcribe|><|{tgt_lang}|><|notimestamps|><|wo_itn|>",
                    "instruction_chat": f"what does the person say{instruction_modifier}?",
                    "output": text,
                } 
            )
        elif task == 'st':
            assert instruction == '', "A fixed instruction should be used in this task."
            output_dict.update(
                {
                    "instruction": f"<|startoftranscript|><|{src_lang}|><|translate|><|{tgt_lang}|><|notimestamps|><|wo_itn|>",
                    "instruction_chat": f"recognize the speech{instruction_modifier}, and translate it into {LANGUAGES[tgt_lang].capitalize()}",
                    "output": text[1],
                    "source_text": text[0],
                    "target_text": text[1],
                }
            )
        elif task in ['slu-sa', 'slu-summ', 'slu-dac', 'dynamic_superb-classification']:
            assert instruction != '', "Please specify the instruction to be used in this task."
            output_dict.update(
                {
                    "instruction_chat": instruction,
                    "output": text,
                }
            )
        elif task in ['slu-sqa']:
            assert separators != [], "Please specify the separator to be used in this task."
            output_dict.update(
                {
                    "instruction_chat": text.split(separators[0])[0].strip(),
                    "output": text.split(separators[1])[1].strip(),
                }
            )
    elif model == 'whisper':
        if task == 'st':
            output_dict.update(
                {
                    "src_lang": LANGUAGES[src_lang].capitalize(),
                    "tgt_lang": LANGUAGES[tgt_lang].capitalize(),
                    "output": text[1],
                    "source_text": text[0],
                    "target_text": text[1],
                } 
            )
        elif task in ['slu-sa', 'slu-summ', 'slu-dac', 'dynamic_superb-classification']:
            output_dict.update(
                {
                    "src_lang": LANGUAGES[src_lang].capitalize(),
                    "tgt_lang": LANGUAGES[tgt_lang].capitalize(),
                    "instruction": instruction,
                    "output": text,
                }
            )
        elif task in ['slu-sqa']:
            output_dict.update(
                {
                    "src_lang": LANGUAGES[src_lang].capitalize(),
                    "tgt_lang": LANGUAGES[tgt_lang].capitalize(),
                    "instruction": text.split(separators[0])[0].strip(),
                    "output": text.split(separators[1])[1].strip(),
                }
            )
        else:
            raise NotImplementedError
    return output_dict

def main(input, split, input_format, task, model, dataset, output_json, dump_dir=None, src_lang=None, tgt_lang=None, specify_src_lang=False, instruction='', separators=''):
    if input_format == 'espnet':
        scps = read_file(os.path.join(input, split, "wav.scp"))
        if task == "asr":
            if src_lang is None: 
                if dataset == 'mls':
                    # Get src_lang and tgt_lang from the name of the input directory
                    src_lang = split.split('_')[1]
                elif dataset == 'librispeech' or dataset == 'tedlium2':
                    src_lang = 'en'
            tgt_lang = src_lang
            texts = read_file(os.path.join(input, split, "text"))
        elif task == "st":
            if src_lang is None or tgt_lang is None: 
                if dataset == 'CoVoST-2':
                    # Get src_lang and tgt_lang from the name of the test split
                    languages = split.split('.')[-1].split('-')
                    languages = [l for l in languages if l in LANGUAGES]
                    assert len(languages) == 2
                    src_lang, tgt_lang = languages
                else:
                    raise Exception("src_lang and tgt_lang should be specified.")
            source_texts = read_file(os.path.join(input, split, f"text.lc.rm.{src_lang}"))
            target_texts = read_file(os.path.join(input, split, f"text.lc.rm.{tgt_lang}"))
            texts = []
            for (id_s, text_s), (id_t, text_t) in zip(source_texts, target_texts):
                assert id_s == id_t, f"File IDs in the source and target text files do not match {id_s}, {id_t}"
                texts.append((id_s, (text_s, text_t)))
        else:
            tgt_lang = src_lang = 'en'
        if "slu" in task:
            texts = read_file(os.path.join(input, split, "text"))
    
        output_list = []
        for (id_scp, audio_path), (id_text, text) in zip(scps, texts):
            assert id_scp == id_text, f"File IDs in the scp and text files do not match {id_scp}, {id_text}"
                
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(input, '/'.join(audio_path.split('/')[-3:]))
            instruction_modifier = ''
            if specify_src_lang:
                assert src_lang is not None
                if model in ['ltu_as', 'salmonn', 'qwen_audio']:
                    instruction_modifier = f' in {LANGUAGES[src_lang].capitalize()}'

            output_dict = format_data(id_scp, audio_path, task, model, dataset, text, src_lang, tgt_lang, instruction=instruction, instruction_modifier=instruction_modifier, separators=separators.split())
            output_list.append(output_dict)
    elif input_format == 'dynamic_superb':
        tgt_lang = src_lang = 'en'
        assert dump_dir is not None, "dump_dir must be specified for dynamic_superb datasets"
        os.makedirs(dump_dir, exist_ok=True)

        from datasets import load_dataset
        from scipy.io.wavfile import write
        import numpy as np

        dset = load_dataset(input, split=split)
        output_list = []
        for item in dset:
            id_ = item['file'].split('.')[0]
            audio, sampling_rate = item['audio']['array'], item['audio']['sampling_rate']
            audio_path = os.path.join(dump_dir, item['file'])
            instruction = item['instruction']
            text = str(item['label'])
            write(audio_path, sampling_rate, audio.astype(np.float32))

            output_dict = format_data(id_, audio_path, task, model, dataset, text, src_lang, tgt_lang, instruction=instruction)
            output_list.append(output_dict)
    else:
        raise NotImplementedError
        
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as of:
        if output_json.endswith(".json"):
            of.write(json.dumps(output_list, indent=4))
        elif output_json.endswith(".jsonl"):
            for line in output_list:
                json.dump(line, of)
                of.write('\n')
        elif output_json.endswith(".tsv"):
            keys = list(output_list[0].keys())
            of.write("\t".join(['id'] + keys))
            of.write('\n')
            for i, line in enumerate(output_list):
                of.write("\t".join([str(i)] + [str(line[k]) for k in keys]))
                of.write('\n')
        else:
            raise NotImplementedError



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the preprocessed espnet-format data that contains wav.scp and text, or HuggingFace dataset ID to DynamicSuperb datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split",
    )
    parser.add_argument(
        "--input_format",
        type=str,
        choices=['espnet', 'dynamic_superb'],
        help="Data format",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=['asr', 'st', 'slu-sa', 'slu-summ', 'slu-sqa', 'dynamic_superb-classification'],
        help="Task to solve",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=['ltu_as', 'salmonn', 'qwen_audio', 'whisper'],
        help="Model to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=['librispeech', 'tedlium2', 'mls', 'CoVoST-2', 'slue-voxceleb', 'slue-ted', 'slue-sqa-5', 'slue-hvb', 'dynamic_superb_sarcasm', 'dynamic_superb_emotion', 'dynamic_superb_dialogue_emotion'],
        help="Dataset used for evaluation",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        help="Path to save the output json file",
    )
    parser.add_argument(
        "--dump_dir",
        type=str,
        help="Directory to save audio data. Only used for Dynamic Superb datasets",
    )
    parser.add_argument(
        "--src_lang",
        type=str,
        choices=LANGUAGES.keys(),
        help="Language of the speech content",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        choices=LANGUAGES.keys(),
        help="Language of the target text",
    )
    parser.add_argument(
        "--specify_src_lang",
        action='store_true',
        help="Whether to specify the source language in the instruction or not",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default='',
        help="Specify the instruction directly.",
    )
    parser.add_argument(
        "--separators",
        type=str,
        default='',
        help="Seperator tokens that are used to extract the instruction and answer \
            from raw text for the slu-sqa task.",
    )
    args = parser.parse_args()
    main(**vars(args))