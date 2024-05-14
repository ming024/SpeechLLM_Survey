from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import fire
from tqdm import tqdm
import json
import os

torch.manual_seed(1234)

def main(
    ckpt_path: str = "",
    eval_json_path: str = "",
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
):
    with open(eval_json_path, 'r') as fp:
        data_json_1 = json.load(fp)

    save_dir = '{:s}/{:s}/{:s}'.format(output_dir, task, eval_dataset)
    result_json_path = '{:s}/result.json'.format(save_dir)
    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as fp:
            result_json = json.load(fp)
    else:
        result_json = []

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="cuda", trust_remote_code=True).eval()

    for i in tqdm(range(len(result_json), len(data_json_1))):
        if 'output' in data_json_1[i]:
            cur_answer = data_json_1[i]["output"]
        else:
            cur_answer = ''
        audio_path = data_json_1[i]["audio_path"]
        audio_id = data_json_1[i]["audio_id"]

        if "Chat" not in ckpt_path:
            prompt = data_json_1[i]["instruction"]        
            query = f"<audio>{audio_path}</audio>{prompt}"
            audio_info = tokenizer.process_audio(query)
            inputs = tokenizer(query, return_tensors='pt', audio_info=audio_info)
            inputs = inputs.to(model.device)
            pred = model.generate(**inputs, audio_info=audio_info)
            output = tokenizer.decode(pred.cpu()[0], skip_special_tokens=False,audio_info=audio_info)
        else:
            prompt = data_json_1[i]["instruction_chat"]
            query = tokenizer.from_list_format([
                {'audio': audio_path}, # Either a local path or an url
                {'text': prompt},
            ])
            output, history = model.chat(tokenizer, query=query, history=None)
        
        result_json.append({'audio_id': audio_id, 'audio_path': audio_path, 'prompt': prompt, 'pred': output, 'ref': cur_answer})
            
        with open(result_json_path, 'w') as fj:
            json.dump(result_json, fj, indent=4)

if __name__ == "__main__":
    fire.Fire(main)