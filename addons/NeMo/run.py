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


import json
import os

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from torch.utils.data import DataLoader

from nemo.collections.multimodal.speechllm.models.speechllm_models import ModularAudioGPTLoRAModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_peft_models import MegatronGPTPEFTModel
from nemo.collections.nlp.models.language_modeling.megatron_gpt_sft_model import MegatronGPTSFTModel
from nemo.collections.nlp.models.nlp_model import NLPModel
from nemo.collections.nlp.parts.megatron_trainer_builder import MegatronTrainerBuilder
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    NLPSaveRestoreConnector,
    PEFTSaveRestoreConnector,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging

mp.set_start_method("spawn", force=True)

@hydra_runner(config_path="examples/multimodel/conf/speechllm", config_name="modularized_speech_gpt_config_eval")
def main(cfg) -> None:
    assert cfg.model.restore_from_path is not None
    assert cfg.model.peft.restore_from_path is not None
    assert cfg.model.peft.restore_from_hparams_path is not None
    megatron_amp_o2 = cfg.model.get("megatron_amp_O2", False)
    with_distributed_adam = False

    trainer = MegatronTrainerBuilder(cfg).create_trainer()
    peft_model_cfg = OmegaConf.to_container(OmegaConf.load(cfg.model.peft.restore_from_hparams_path).cfg)
    peft_model_cfg = OmegaConf.create(peft_model_cfg)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(peft_model_cfg):
        # update the model config of the trained model with params we want to set at inference time.
        peft_model_cfg.precision = cfg.trainer.precision
        peft_model_cfg.data.test_ds = cfg.model.data.test_ds
        peft_model_cfg.activations_checkpoint_granularity = None
        peft_model_cfg.activations_checkpoint_method = None
        if peft_model_cfg.get("use_flash_attention", False):
            peft_model_cfg.use_flash_attention = cfg.model.use_flash_attention
        if cfg.model.get("seq_len_interpolation_factor", None) is not None:
            peft_model_cfg["seq_len_interpolation_factor"] = cfg.model.seq_len_interpolation_factor
        if hasattr(cfg.model, "pretrained_audio_model"):
            peft_model_cfg.pretrained_audio_model = cfg.model.pretrained_audio_model

    if '\\' in cfg.model.peft.restore_from_path:
        cfg.model.peft.restore_from_path = cfg.model.peft.restore_from_path.replace('\\', '')
    # attempting to load a ckpt peft model.
    if cfg.model.peft.restore_from_ckpt_name:
        ckpt_name = cfg.model.peft.restore_from_ckpt_name
    else:
        ckpt_name = "model_weights.ckpt"
    save_restore_connector = PEFTSaveRestoreConnector(
        peft_model_nemo_path=None,
        peft_model_ckpt_path=cfg.model.peft.restore_from_path,
        peft_model_ckpt_name=ckpt_name,
    )

    if os.path.isdir(cfg.model.restore_from_path):
        save_restore_connector.model_extracted_dir = cfg.model.restore_from_path

    model = ModularAudioGPTLoRAModel.restore_from(
        restore_path=cfg.model.restore_from_path,
        trainer=trainer,
        override_config_path=peft_model_cfg,
        save_restore_connector=save_restore_connector,
    )
    config = OmegaConf.to_container(cfg.inference, resolve=True)
    model.set_inference_config(config)
    model.freeze()
    _test_ds = model._build_dataset(peft_model_cfg.data.test_ds, is_train=False)
    if isinstance(_test_ds, list):
        _test_ds = _test_ds[0]

    request_dl = DataLoader(
        dataset=_test_ds, batch_size=peft_model_cfg.data.test_ds.global_batch_size, collate_fn=_test_ds.collate_fn,
    )

    response = trainer.predict(model, request_dl)

    if model.global_rank == 0:
        result_json = []
        for batch in response:
            batch_sentences = [s for s in batch['sentences']]
            batch_tokens = [s for s in batch['tokens']]
            batch_contexts = [s for s in batch['inputs']]
            batch_labels = [s for s in batch['labels']]
            batch_preds = [s for s in batch['preds']]
            batch_metadata = [s for s in batch['metadata']]
            for i in range(len(batch_sentences)):
                result_json.append({'audio_id': batch_metadata[i]["audio_id"], 'audio_path': batch_metadata[i]["audio_filepath"], 'prompt': batch_contexts[i], 'input': "", 'pred': batch_preds[i], 'ref': batch_labels[i]})
            
    with open(cfg.inference.outfile_path, 'w') as fj:
        json.dump(result_json, fj, indent=4)


if __name__ == "__main__":
    main()
