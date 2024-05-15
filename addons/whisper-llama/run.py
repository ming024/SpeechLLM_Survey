import json
import os
import fire
import torch
import transformers
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, pipeline


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def main(
    eval_json_path: str = "",
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
):
    whisper_model_id = "openai/whisper-large-v3"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        whisper_model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    whisper_model.to(device)
    whisper_processor = AutoProcessor.from_pretrained(whisper_model_id)

    whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=whisper_processor.tokenizer,
        feature_extractor=whisper_processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=1,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    llama_model_id = "meta-llama/Llama-2-7b-chat-hf"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    llama_pipeline = transformers.pipeline(
        "text-generation", 
        model=llama_model_id,
        tokenizer=llama_tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        eos_token_id=llama_tokenizer.eos_token_id
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

    for i in tqdm(range(len(result_json), len(data_json_1))):
        if 'output' in data_json_1[i]:
            cur_answer = data_json_1[i]["output"]
        else:
            cur_answer = ''
        audio_path = data_json_1[i]["audio_path"]
        audio_id = data_json_1[i]["audio_id"]

        src_lang = data_json_1[i]["src_lang"]
        tgt_lang = data_json_1[i]["tgt_lang"]
        instruction = data_json_1[i].get("instruction", None)

        transcript = whisper_pipe(audio_path, generate_kwargs={"language": src_lang, "task": "transcribe"})["text"]

        if task == "st":
            template = """Translate the following sentence from {src_lang} to {tgt_lang}:
            ```{text}```
            TRANSLATED SENTENCE:
            """
            prompt = template.format(src_lang=src_lang, tgt_lang=tgt_lang, text=transcript)
        elif task == "slu-sa":
            instruction = instruction.replace("speech", "text below")
            template = f"""{instruction}

            ### TEXT: 
            {transcript}

            ### SENTIMENT:
            """
            prompt = template.format(instruction=instruction, text=transcript)
        elif task == "slu-summ":
            instruction = instruction.replace("speech", "document below")
            template = f"""{instruction}

            ### DOCUMENT:
            {transcript}

            ### SUMMARY:
            """
            prompt = template.format(instruction=instruction, text=transcript)
        elif task == "slu-sqa":
            template = f"""Please answer the question based on the information provided in the document below.

            ### DOCUMENT:
            {transcript}

            ### QUESTION:
            {instruction}

            ### ANSWER:
            """
            prompt = template.format(instruction=instruction, text=transcript)

        output = llama_pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id=llama_tokenizer.eos_token_id,
            pad_token_id=llama_tokenizer.eos_token_id,
            return_full_text=False,
            temperature=.9,
            top_p=0.95,
        )[0]['generated_text']
        
        result_json.append({'audio_id': audio_id, 'audio_path': audio_path, 'transcript': transcript, 'pred': output, 'ref': cur_answer})
            
        with open(result_json_path, 'w') as fj:
            json.dump(result_json, fj, indent=4)

if __name__ == "__main__":
    fire.Fire(main)