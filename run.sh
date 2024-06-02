#!/bin/bash
model=$1
data_dir=$2 # eval_data/salmonn
dset=$3
specify_src_lang=$4
espnet_root=/ocean/projects/cis210027p/cchien1/espnet/
data_file_format=json
result_file_format=json



# Model specific setups
if [[ "$model" == "salmonn_7b" ]]; then
    source ~/.bashrc
    conda activate SALMONN
    beats_path=models/SALMONN/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    ckpt_path=models/SALMONN/ckpt/salmonn_7b_v0.pth
    vicuna_path=lmsys/vicuna-7b-v1.5
    use_low_resource=false
elif [[ "$model" == "salmonn_13b" ]]; then
    source ~/.bashrc
    conda activate SALMONN
    beats_path=models/SALMONN/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
    ckpt_path=models/SALMONN/ckpt/salmonn_v1.pth
    vicuna_path=lmsys/vicuna-13b-v1.1
    use_low_resource=true
elif [[ "$model" == "qwen_audio" ]]; then
    source ~/.bashrc
    conda activate Qwen_Audio
    ckpt_path=Qwen/Qwen-Audio
elif [[ "$model" == "qwen_audio_chat" ]]; then
    source ~/.bashrc
    conda activate Qwen_Audio
    ckpt_path=Qwen/Qwen-Audio-Chat
elif [[ "$model" == "ltu_as" ]]; then
    source ~/.bashrc
    conda activate LTU_AS
    vicuna_path=./models/ltu/pretrained_mdls/vicuna_ltuas/
    eval_mdl_path=./models/ltu/pretrained_mdls/ltuas_long_noqa_a6.bin
    whisper_feat_mdl_path=./models/ltu/pretrained_mdls/large-v1.pt
elif [[ "$model" == "whisper" ]]; then
    source ~/.bashrc
    conda activate whisper
elif [[ "$model" == "whisper-llama" ]]; then
    source ~/.bashrc
    conda activate whisper-llama
fi



# Data and output directories
if $specify_src_lang; then
    data_json_dir=${data_dir}/specify_src_lang/
    espnet_res_dir=${espnet_root}/egs2/${model}_specify_src_lang
    output_dir=eval_results/${model}_specify_src_lang
else
    data_json_dir=${data_dir}/not_specify_src_lang/
    espnet_res_dir=${espnet_root}/egs2/${model}
    output_dir=eval_results/${model}
fi

# Through this symbolic link to espnet_res_dir, we can use the evaluation functions built in espnet
mkdir -p $espnet_res_dir
if [ ! -e $output_dir ]; then
    ln -s $espnet_res_dir $output_dir
fi



# Data specific variables
save_ref=false
if [[ "$dset" == "librispeech" ]]; then
    splits="test_clean test_other"
    espnet_task=asr2
    task=asr
    tgt_lang="en"
elif [[ "$dset" == "tedlium2" ]]; then
    splits="test "
    espnet_task=asr1
    task=asr
    tgt_lang="en"
elif [[ "$dset" == "mls" ]]; then
    splits="mls_fr_test mls_nl_test mls_pt_test mls_de_test mls_es_test mls_it_test mls_pl_test"
    espnet_task=asr2
    task=asr
    tgt_lang="multi"
elif [[ "$dset" == "CoVoST-2" ]]; then
    splits="test.ca-en test.en-ar test.en-cy test.en-et test.en-id test.en-lv test.en-sl test.en-ta test.en-zh-CN test.fr-en test.zh-CN-en test.de-en test.en-ca test.en-de test.en-fa test.en-ja test.en-mn test.en-sv-SE test.en-tr test.es-en test.ja-en"
    espnet_task=st1
    task=st
elif [[ "$dset" == "slue-voxceleb" ]]; then
    splits="test.0 test.1 test.2 test.3 test.4 test.5"
    espnet_task=slu1
    task=slu-sa
elif [[ "$dset" == "slue-ted" ]]; then
    splits="test.0 test.1 test.2"
    espnet_task=slu1
    task=slu-summ
elif [[ "$dset" == "slue-sqa-5" ]]; then
    splits="test"
    espnet_task=slu1
    task=slu-sqa
    # We need to save reference text for SQA evaluation
    save_ref=true
fi


for split in ${splits}; do
    if  [[ "$split" == *"$dset"* ]]; then
        output_name="${split}"
    else
        output_name="${dset}_${split}"
    fi

    data_json_path=${data_json_dir}/${output_name}.${data_file_format}
    mkdir -p ${output_dir}/${task}/${output_name}


    echo "Evaluating ${model} on ${output_name}"
    echo "Running inference..."

    if [[ "$model" == "salmonn_7b" ]] || [[ "$model" == "salmonn_13b" ]]; then
        python3 models/SALMONN/run.py --ckpt_path $ckpt_path --eval_dataset $output_name --task $task --output_dir $output_dir --vicuna_path $vicuna_path --beats_path $beats_path --use_low_resource $use_low_resource --eval_json_path $data_json_path
    elif [[ "$model" == *"qwen"* ]]; then
        python3 models/Qwen-Audio/run.py --ckpt_path $ckpt_path --eval_dataset $output_name --task $task --output_dir $output_dir --eval_json_path $data_json_path
    elif [[ "$model" == "ltu_as" ]]; then
        python3 models/ltu/src/ltu_as/run.py  --eval_dataset $output_name --task $task --output_dir $output_dir --base_model $vicuna_path --eval_mdl_path $eval_mdl_path --eval_json_path $data_json_path --whisper_feat_mdl_path $whisper_feat_mdl_path
    elif [[ "$model" == "whisper" ]]; then
        python3 models/whisper/run.py --eval_dataset $output_name --task $task --output_dir $output_dir --eval_json_path $data_json_path --save_ref $save_ref
    elif [[ "$model" == "whisper-llama" ]]; then
        python3 models/whisper-llama/run.py --eval_dataset $output_name --task $task --output_dir $output_dir --eval_json_path $data_json_path
    fi


    echo "Post-processing LLM outputs"
    if [[ "$model" == "whisper-llama" ]]; then
        source ~/.bashrc
        conda activate whisper-llama
        python3 post_process_text.py --eval_dataset $output_name --task $task  --output_dir $output_dir  --save_ref $save_ref --eval_data_file ${data_json_dir}/${output_name}.json --result_file_format $result_file_format --use_llama true
    else
        source ~/.bashrc
        conda activate whisper-llama
        python3 post_process_text.py --eval_dataset $output_name --task $task  --output_dir $output_dir  --save_ref $save_ref --result_file_format $result_file_format --use_llama true
    fi


    echo "Scoring results"
    if [[ "$task" == "asr" ]]; then
        score_opts=(--dataset_name $dset --task $espnet_task --test_sets "${split}" --tgt_case ts --tgt_lang $tgt_lang)
    elif [[ "$task" == "st" ]]; then
        src_tgt=$(echo "$split" | sed 's/test.//g' | sed 's/-CN//g' | sed 's/-SE//g')
        src_lang="$(cut -d'-' -f1 <<<${src_tgt})"
        tgt_lang="$(cut -d'-' -f2 <<<${src_tgt})"
        score_opts=(--dataset_name $dset --src_lang $src_lang --tgt_lang $tgt_lang --test_sets $split)
    elif [[ "$task" == *"slu"* ]]; then
        score_opts=(--dataset_name $dset  --task $task --test_sets "${split}")
    fi

    if [ ! -e "${output_dir}/${task}/run.sh" ]; then
        ln -s ${PWD}/eval_results/run_$(cut -d'-' -f1 <<<${task}).sh ${output_dir}/${task}/run.sh
    fi
    if [ ! -e "${output_dir}/${task}/eval.sh" ]; then
        ln -s ${PWD}/eval_results/eval_$(cut -d'-' -f1 <<<${task}).sh ${output_dir}/${task}/eval.sh
    fi
    cd ${output_dir}/${task}/
    ./run.sh "${score_opts[@]}"
    cd ../../../
done