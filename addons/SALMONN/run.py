import torch
import fire
from model import SALMONN
from tqdm import tqdm
import json
import os

def main(
    ckpt_path: str = "",
    whisper_path: str = "openai/whisper-large-v2",
    beats_path: str = "beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
    vicuna_path: str = "lmsys/vicuna-7b-v1.5",
    eval_json_path: str = "",
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
    use_low_resource: bool = False,
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

    model = SALMONN(
        ckpt=ckpt_path,
        whisper_path=whisper_path,
        beats_path=beats_path,
        vicuna_path=vicuna_path,
        low_resource=use_low_resource,
    )
    model.to("cuda:0")
    model.eval()

    for i in tqdm(range(len(result_json), len(data_json_1))):
        if 'output' in data_json_1[i]:
            cur_answer = data_json_1[i]["output"]
        else:
            cur_answer = ''
        audio_path = data_json_1[i]["audio_path"]
        audio_id = data_json_1[i]["audio_id"]

        prompt = data_json_1[i]["instruction"]

        # for environment with cuda>=117
        with torch.cuda.amp.autocast(dtype=torch.float16):
            output = model.generate(audio_path, prompt=prompt)[0]
        
        result_json.append({'audio_id': audio_id, 'audio_path': audio_path, 'prompt': prompt, 'input': "", 'pred': output, 'ref': cur_answer})
            
        with open(result_json_path, 'w') as fj:
            json.dump(result_json, fj, indent=4)

if __name__ == "__main__":
    fire.Fire(main)