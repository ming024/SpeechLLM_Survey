import os
import fire
import json
import re
import random
import numpy as np
import csv
import pandas as pd

def read_result_file(file, file_format, eval_data_tsv):
    if file_format == "json":
        with open(file, 'r') as f:
            data = json.load(f)
        return data
    elif file_format == "txt":
        # For WavLLM outputs
        data = []
        with open(file, 'r') as f:
            for line1 in f:
                try:
                    line2 = next(f)
                except:
                    break
                data.append(
                    {
                        "ref": line1.split('\t')[1].strip('\n'),
                        "pred": line2.split('\t')[2].strip('\n').strip('\'').strip('\"'),
                    }
                )
        
        # Read audio_id from eval_data_tsv since it is not included in the output format of WavLLM
        assert eval_data_tsv != ""
        eval_data = pd.read_csv(eval_data_tsv, sep='\t', quoting=csv.QUOTE_NONE).transpose().to_dict()
        eval_data = [eval_data[i] for i in range(len(eval_data))]
        assert len(data) == len(eval_data)
        for i in range(len(eval_data)):
            assert data[i]["ref"] == eval_data[i]["tgt_text"]
            data[i]["audio_id"] = eval_data[i]["audio_id"]
        
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

def extract_class_label(text, classes):
    if classes is None:
        return text

    pred = np.zeros((len(classes), ))
    for i, c in enumerate(classes):
        if c.lower() in text:
            pred[i] = 1
    
    if np.sum(pred) == 0:
        # No valid option in the text
        return random.choice(classes)
    elif np.sum(pred) == 1:
        # Only one valid option seen in the text
        return classes[np.argmax(pred)]
    else:
        # Multiple valid options seen in the text so we choose one of them by random
        return random.choice([classes[i] for i, p in enumerate(pred) if p > 0 ])

def extract_class_label_ltu_as(text, classes):
    if classes is None:
        return text

    # Remove text in parenthesis since they can be misleading
    text = re.sub(r"[\(].*?[\)]", "", text)

    possible_prefixes = [
        "speech sentiment is",
        "the speech emotion is",
        "the sentiment is",
        "classify the sentiment as",
        "the sentiment of the speech is",
        "the sentiment of the speech is weakly",
        "the speech is",
        "the sentiment of this speech is",
        "",
    ]
    for prefix in possible_prefixes:
        if text.startswith(prefix):
            pred_label = re.sub(r"[^A-Za-z]", "", text.split(" ")[len(prefix.split())]).capitalize()
            if pred_label in classes:
                return pred_label

    # The model does not make a decision 
    invalid_prefixes = [
        "the given information does not allow",
    ]
    for prefix in invalid_prefixes:
        if text.startswith(prefix):
            return random.choice(classes)

    # Counting the class labels
    pred = np.zeros((len(classes), ))
    for i, c in enumerate(classes):
        if c.lower() in text:
            pred[i] = 1

    if np.sum(pred) == 0:
        return random.choice(classes)
    elif np.sum(pred) == 1:
        return classes[np.argmax(pred)]
    else:
        # Multiple valid options seen in the text so we choose one of them by random
        return random.choice([classes[i] for i, p in enumerate(pred) if p > 0 ])

def main(
    task: str = "",
    eval_dataset: str = "",
    output_dir: str = "",
    save_ref: bool = False,
    result_file_format: str = "json",
    eval_data_tsv: str = "",
):
    result_file_path = f'{output_dir}/{task}/{eval_dataset}/result.{result_file_format}'
    data = read_result_file(result_file_path, result_file_format, eval_data_tsv)

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
            # For Qwen-Audio only
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
    elif "dynamic_superb_sarcasm" in eval_dataset:
        classes = ["True", "False"]
    elif "dynamic_superb_emotion" in eval_dataset:
        classes = ["Anger", "Disgust", "Sadness", "Joy", "Neutral", "Surprise", "Fear"]
    elif "dynamic_superb_dialogue_emotion" in eval_dataset:
        classes = ["Anger", "Disgust", "Fear", "Sadness", "Happiness", "Surprise", "No emotion"]

    output_texts = []
    ref_texts = []
    for i in range(len(data)):
        output_text = postprocess_text(data[i]["pred"], separators, regex_patterns)
        if 'ltu_as' in output_dir:
            output_text = extract_class_label_ltu_as(output_text, classes)
        else:
            output_text = extract_class_label(output_text, classes)

        # Extract audio_id from path
        if '/' in data[i]["audio_id"]:
            data[i]["audio_id"] = data[i]["audio_id"].split('/')[-1].split('.')[0]

        output_texts.append(f'{data[i]["audio_id"]} {output_text}')
        ref_texts.append(f'{data[i]["audio_id"]} {data[i]["ref"]}')

    with open(f'{output_dir}/{task}/{eval_dataset}/text', 'w') as ft:
        ft.write('\n'.join(output_texts))
    if save_ref:
        with open(f'{output_dir}/{task}/{eval_dataset}/ref_text', 'w') as ft:
            ft.write('\n'.join(ref_texts))

if __name__ == "__main__":
    fire.Fire(main)
