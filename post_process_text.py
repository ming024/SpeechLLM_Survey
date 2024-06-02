import os
import fire
import json
import re
import random
import numpy as np
import csv
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline

llama_model_id = llama_tokenizer = llama_pipeline = None
random_counts = 0
llama_counts = 0
rescored_texts = []

def read_result_file(file, file_format, eval_data_file):
    if file_format == "json":
        with open(file, 'r') as f:
            data = json.load(f)
        if eval_data_file:
            with open(eval_data_file, 'r') as f:
                eval_data = json.load(f)
        else:
            eval_data = None

    if eval_data is not None:
        assert len(data) == len(eval_data)
        for i in range(len(eval_data)):
            if "audio_id" not in data[i]:
                data[i]["audio_id"] = eval_data[i]["audio_id"]
            if "prompt" not in data[i]:
                if "prompt" in eval_data[i]:
                    data[i]["prompt"] = eval_data[i]["prompt"]
                else:
                    data[i]["prompt"] = eval_data[i]["instruction"]
        
    return data

def postprocess_text(text, separators, regex_patterns):
    text = text.lower().lstrip()
    
    for sep in separators:
        if isinstance(sep, tuple):
            # Get the split_id-th element after splitting the text with "sep"
            sep, split_id = sep
            split_list = [e for e in re.split(sep, text) if e != '']
            text = split_list[split_id] if len(split_list) > 0 else ''
        else:
            # If split_id not specified, get the first element
            if sep in text:
                text = text.partition(sep)[0]

    for r1, r2 in regex_patterns:
        text = re.sub(r1, r2, text).lower()
    return text.strip()

def extract_class_label(text, classes, prompt, use_llama=False):
    if classes is None:
        return text

    global random_counts, llama_counts, rescored_texts

    pred_label = None
    while True:
        pred = np.zeros((len(classes), ))
        for i, c in enumerate(classes):
            if c.lower() in text:
                pred[i] = 1
    
        if np.sum(pred) == 0:
            # No valid option in the text
            if use_llama:
                llama_counts += 1
                rescored_texts.append((text, None))
                text = extract_class_label_llama(text, classes, prompt)
                use_llama = False
            else:
                random_counts += 1
                pred_label = f"{random.choice(classes)}\t(Random)"
        elif np.sum(pred) == 1:
            # Only one valid option seen in the text
            pred_label = classes[np.argmax(pred)]
        else:
            if use_llama:
                llama_counts += 1
                rescored_texts.append((text, None))
                text = extract_class_label_llama(text, classes, prompt)
                use_llama = False
            else:
                # Multiple valid options seen in the text so we choose one of them by random
                random_counts += 1
                pred_label = f"{random.choice([classes[i] for i, p in enumerate(pred) if p > 0 ])}\t(Random)"
        
        if pred_label:
            if rescored_texts and rescored_texts[-1][1] is None:
                rescored_texts[-1] = (rescored_texts[-1][0], pred_label)
            return pred_label.split('\t')[0]

def extract_class_label_llama(text, classes, prompt):
    global llama_model_id, llama_tokenizer, llama_pipeline
    if not llama_model_id:
        llama_model_id = "meta-llama/Llama-2-7b-chat-hf"
        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
        llama_pipeline = pipeline(
            "text-generation", 
            model=llama_model_id,
            tokenizer=llama_tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            max_new_tokens=10,
            eos_token_id=llama_tokenizer.eos_token_id
        )

    template = f"""### QUESTION: Identify the presence of sarcasm or ironic expressions in the speech. The answer could be true or false.
    ### VALID OPTIONS: "True", "False"
    ### USER RESPONSE: yes.
    ### SELECTED OPTION: "True"


    ### QUESTION: Clarify the emotion of the dialogue. The answer could be anger, disgust, fear, sadness, happiness, surprise, or no emotion.
    ### VALID OPTIONS: "Anger", "Disgust", "Fear", "Sadness", "Happiness", "Surprise", "No emotion"
    ### USER RESPONSE: The emotion of the dialogue is not specified.
    ### SELECTED OPTION: "No emotion"


    ### QUESTION: Identify the presence of sarcasm or ironic expressions in the speech. The answer could be true or false.
    ### VALID OPTIONS: "True", "False"
    ### USER RESPONSE: Based on the voice, it sounds like this person is not being sarcastic or ironic in the speech, saying, "As you may know, I've been experimenting with elevated anxiety levels." without a tone of sarcasm.
    ### SELECTED OPTION: "False"


    QUESTION: Listen to the audio and categorize the emotion. The answer could be anger, disgust, fear, sadness, happiness, surprise, or no emotion.
    ### VALID OPTIONS: "Anger", "Disgust", "Fear", "Sadness", "Happiness", "Surprise", "No emotion"
    ### USER RESPONSE: Based on the voice, it sounds like this person is very angry.
    ### SELECTED OPTION: "Anger"
    

    QUESTION: {prompt}
    ### VALID OPTIONS: {', '.join([f'"{c}"' for c in classes])}
    ### USER RESPONSE: {text}
    ### SELECTED OPTION: """
    prompt = template.format(prompt=prompt, classes=classes, text=text)
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

    return output.lower()

def main(
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
    save_ref: bool = False,
    result_file_format: str = "json",
    eval_data_file: str = "",
    use_llama: bool = True,
):
    result_file_path = f'{output_dir}/{task}/{eval_dataset}/result.{result_file_format}'
    data = read_result_file(result_file_path, result_file_format, eval_data_file)

    classes = None
    regex_patterns = []
    separators = ["\n", "\r"]

    # For LTU_AS only
    if 'ltu_as' in output_dir:
        separators += [("spoken text:", -1)]
    
    # For Qwen-Audio only
    if 'qwen_audio' in output_dir:
        if 'qwen_audio_chat' in output_dir:
            regex_patterns += [
                (r".*the person says.*: \"(.*)\".*", r"\1"),
                (r".*translated into.*is \"(.*)\".*", r"\1"),
            ]
        else:
            # For Qwen-Audio only, remove language tags
            if "asr" in task or 'st' in task:
                separators += [(r'<.*>', -1)]

    # For any model
    if 'librispeech' in eval_dataset:
        # Only keep word characters, digits, and apostrophes
        # Replace hyphens with spaces
        regex_patterns += [
            (r'[^\w\d\s\'-]+', ''),
            (r'[-]+', ' '),
        ]
    elif "tedlium2" in eval_dataset or  "mls" in eval_dataset:
        # Only keep word characters, digits, apostrophes, and hyphens
        regex_patterns += [
            (r'[^\w\d\s\'-]+', ''),
        ]
    else:
        pass

    if "slue-voxceleb" in eval_dataset:
        classes = ["Positive", "Neutral", "Negative"]

    output_texts = []
    ref_texts = []
    for i in tqdm(range(len(data))):
        output_text = postprocess_text(data[i]["pred"], separators, regex_patterns)
        prompt = data[i]["prompt"]
        output_text = extract_class_label(output_text, classes, prompt, use_llama)

        # Extract audio_id from path
        if '/' in data[i]["audio_id"]:
            data[i]["audio_id"] = data[i]["audio_id"].split('/')[-1].split('.')[0]

        output_texts.append(f'{data[i]["audio_id"]} {output_text.strip().capitalize()}')
        ref_texts.append(f'{data[i]["audio_id"]} {data[i]["ref"].strip().capitalize()}')

    with open(f'{output_dir}/{task}/{eval_dataset}/text', 'w') as ft:
        ft.write('\n'.join(output_texts))
    if save_ref:
        with open(f'{output_dir}/{task}/{eval_dataset}/ref_text', 'w') as ft:
            ft.write('\n'.join(ref_texts))
    
    with open(f'{output_dir}/{task}/{eval_dataset}/rescored_texts', 'w') as ft:
        global random_counts, llama_counts, rescored_texts
        ft.write(f"Randomly assigned class labels: {random_counts}\n")
        ft.write(f"Class labels assigned by Llama: {llama_counts - random_counts}\n")
        ft.write("\n".join(["\n".join(line) for line in rescored_texts]))

if __name__ == "__main__":
    fire.Fire(main)
