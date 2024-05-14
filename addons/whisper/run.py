import json
import os
import torch
import fire
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def main(
    eval_json_path: str = "",
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
    save_ref: bool = False,
):
    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    with open(eval_json_path, 'r') as fp:
        data_json_1 = json.load(fp)

    save_dir = '{:s}/{:s}/{:s}'.format(output_dir, task, eval_dataset)
    result_json_path = '{:s}/result.json'.format(save_dir)
    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as fp:
            result_json = json.load(fp)
    else:
        result_json = []

    output_texts = []
    ref_texts = []
    for i in tqdm(range(len(result_json), len(data_json_1))):
        if 'output' in data_json_1[i]:
            cur_answer = data_json_1[i]["output"]
        else:
            cur_answer = ''
        audio_path = data_json_1[i]["audio_path"]
        audio_id = data_json_1[i]["audio_id"]

        src_lang = data_json_1[i]["src_lang"]
        tgt_lang = data_json_1[i]["tgt_lang"]

        if src_lang == tgt_lang:
            output = pipe(audio_path, generate_kwargs={"language": tgt_lang, "task": "transcribe"})["text"]
        elif tgt_lang.lower() == "english":
            output = pipe(audio_path, generate_kwargs={"language": tgt_lang, "task": "translate"})["text"]
        else:
            output = pipe(audio_path, generate_kwargs={"language": tgt_lang, "task": "transcribe"})["text"]
        
        result_json.append({'audio_id': audio_id, 'audio_path': audio_path, 'pred': output, 'ref': cur_answer})
        output_texts.append(f'{data_json_1[i]["audio_id"]} {output}')
        ref_texts.append(f'{data_json_1[i]["audio_id"]} {cur_answer}')
            
        with open(result_json_path, 'w') as fj:
            json.dump(result_json, fj, indent=4)

    with open(f'./{output_dir}/{task}/{eval_dataset}/text', 'w') as ft:
        ft.write('\n'.join(output_texts))
    if save_ref:
        with open(f'./{output_dir}/{task}/{eval_dataset}/ref_text', 'w') as ft:
            ft.write('\n'.join(ref_texts))

if __name__ == "__main__":
    fire.Fire(main)