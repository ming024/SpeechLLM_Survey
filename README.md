# Reproducing the SpeechLLM Benchmarking Results in the Paper "Spoken Language Models: A Survey and Empirical Comparison"

## Dependencies
Our codebase relies on ESP-Net for the evaluation of ASR, ST, and SLU tasks. Please follow this [link](https://github.com/espnet/espnet) to install ESP-Net step-by-step.

## Add submodules of the surveyed SpeechLLMs
```
git submodule add git@github.com:YuanGongND/ltu.git ./models/ltu
git submodule add git@github.com:zhehuaichen/NeMo.git ./models/NeMo
git submodule add git@github.com:QwenLM/Qwen-Audio.git ./models/Qwen-Audio
git submodule add git@github.com:bytedance/SALMONN.git ./models/SALMONN
```

After adding the submodules, you should follow the instructions in the individual codebases to create conda environments for the individual models.

Also, run
```
pip install requriements.txt
```
in ``models/whisper`` and ``models/whisper-llama``.

## Put addons files in the submodule folders

Add files in ``addons/*/`` to ``models/*/``

## Data pre-processing
Please follow [ESP-Net data pre-processing steps](https://github.com/espnet/espnet/tree/master/egs2) to pre-process the speech data into Kaldi formats (i.e., with the .scp files).

## Change environment-specific setups
Change ``espnet_root`` in ``run.sh`` to the path of the installed ESP-Net package.

Change ``_data`` in ``eval_results/eval_xxx.sh``  to the path of your preprocessed data (in ESP-Net format).

## Run benchmarking
Run
```
. prep_data.sh $model_name $dataset_name
```
to prepare data JSON files (which contain the audio paths, text prompts, etc.) for evaluation.

Then, run
```
. run.sh $model_name eval_data/$model_name $dataset_name $specify_src_lang
```
for benchmarking the performance of the SpeechLLM of interest.
The variable ``$specify_src_lang`` should be set to ``True`` only in ST evaluations. 
When this option is turned on, the source language will be specified in the text prompts. 
See section 5 in the paper for more information about the experiment setup.

After running the code, the results should be saved in ``./eval_results``

<!-- ## Add New Model
- Add new folder in models/
- Modify prep_data.sh and prep_data.py
- Add models/xxx/run.py
- Add run_xxx.sh in root directory. We may take previous run_xxx.sh as references
- Modify post_process_text.py

## Add New Task
- Modify prep_data.sh and prep_data.py
- Modify post_process_text.py
- Add eval_results/run_xxxx.sh and eval_results/eval_***.sh
- Modify the scoring session in run.sh -->