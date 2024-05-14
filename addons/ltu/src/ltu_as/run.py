import os
import fire
import json
import torch
import time
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from utils.prompter import Prompter
import numpy as np
import skimage.measure
import whisper_at
from whisper.model import Whisper, ModelDimensions
from tqdm import tqdm
import re

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_text_model = whisper_at.load_model("large-v2", device='cuda:0')

def load_whisper(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    dims = ModelDimensions(**checkpoint["dims"])
    whisper_feat_model = Whisper(dims)
    whisper_feat_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    whisper_feat_model.to('cuda:0')
    return whisper_feat_model

def convert_params_to_float32(model):
    for name, param in model.named_parameters():
        if "audio_encoder" in name and "ln" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()

text_cache = {}
def load_audio_trans(filename, whisper_feat_model):
    global text_cache
    if filename not in text_cache:
        result = whisper_text_model.transcribe(filename)
        text = result["text"].lstrip()
        text_cache[filename] = text
    else:
        text = text_cache[filename]
        print('using asr cache')
    _, audio_feat = whisper_feat_model.transcribe_audio(filename)
    audio_feat = audio_feat[0]
    audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
    audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
    audio_feat = audio_feat[1:]  # skip the first layer
    audio_feat = torch.FloatTensor(audio_feat)
    return audio_feat, text

def main(
    load_8bit: bool = False,
    base_model: str = "../../pretrained_mdls/vicuna_ltuas",
    espnet_root: str = "/ocean/projects/cis210027p/cchien1/espnet/egs2/ltu_as",
    eval_mdl_path: str = "../../pretrained_mdls/ltuas_long_noqa_a6.bin",
    whisper_feat_mdl_path: str = "../../pretrained_mdls/large-v1.pt",
    eval_json_path: str = "",
    prompt_template: str = "alpaca_short",
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
    eval_mode: str = "joint"
):
    assert eval_mode in ['asr_only', 'audio_only', 'joint']

    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16) #, torch_dtype=torch.float16

    convert_params_to_float32(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    whisper_feat_model = load_whisper(whisper_feat_mdl_path)

    temp, top_p, top_k = 0.1, 0.95, 500
    if eval_mdl_path != 'vicuna':
        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        miss, unexpect = model.load_state_dict(state_dict, strict=False)
        print('unexpect', unexpect)

    model.is_parallelizable = True
    model.model_parallel = True

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.eval()

    # all these json file can be downloaded from https://www.dropbox.com/scl/fo/o91k6cnwqft84tgmuotwg/h?rlkey=6bnjobvrbqbt4rqt3f1tgaeb8&dl=0
    # you will need to prepare whisper feature by yourself, note please convert all audios to 16khz

    with open(eval_json_path, 'r') as fp:
        data_json_1 = json.load(fp)

    save_dir = '{:s}/{:s}/{:s}'.format(output_dir, task, eval_dataset)
    result_json_path = '{:s}/result.json'.format(save_dir)
    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as fp:
            result_json = json.load(fp)
    else:
        result_json = []

    for i in tqdm(range(len(result_json), len(data_json_1))):
        if 'output' in data_json_1[i]:
            cur_answer = data_json_1[i]["output"]
        else:
            cur_answer = ''
        audio_path = data_json_1[i]["audio_path"]
        audio_id = data_json_1[i]["audio_id"]

        instruction = data_json_1[i]["instruction"]

        begin_time = time.time()

        if audio_path != 'empty':
            cur_audio_input, cur_input = load_audio_trans(audio_path, whisper_feat_model)
        else:
            print('loading audio error')
            cur_audio_input, cur_input = None, ""
        cur_input = data_json_1[i]["input"].replace("Spoken text: ", "")

        prompt = prompter.generate_prompt(instruction, cur_input)

        # print('Input prompt: ', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        if torch.cuda.is_available() == False:
            pass
        else:
            cur_audio_input = cur_audio_input.unsqueeze(0).half().to(device)  # .half().to(device)
            # print(cur_audio_input.dtype)
        
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.1,
            max_new_tokens=400,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            num_return_sequences=1
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids.to(device),
                audio_input=cur_audio_input,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=400,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        output = output[5:-4]
        end_time = time.time()
        # print('eclipse time: ', end_time-begin_time, ' seconds.')

        result_json.append({'audio_id': audio_id, 'audio_path': audio_path, 'prompt': instruction, 'input': cur_input, 'pred': output[len(prompt)+1:], 'ref': cur_answer})
        
        with open(result_json_path, 'w') as fj:
            json.dump(result_json, fj, indent=4)
            
if __name__ == "__main__":
    fire.Fire(main)
