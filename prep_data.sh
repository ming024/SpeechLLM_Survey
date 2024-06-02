model=$1
dset=$2
instructions=
opts=


# Data formats depend on models
output_file_format=json



# Dataset specific variables
if [[ "$dset" == "librispeech" ]]; then
    splits="test_clean test_other"
    espnet_task=asr2
    task=asr
    input_format=espnet
    input_path=/ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/${espnet_task}/dump/audio_raw/
elif [[ "$dset" == "tedlium2" ]]; then
    splits="test "
    espnet_task=asr1
    task=asr
    input_format=espnet
    input_path=/ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/${espnet_task}/dump/raw/
elif [[ "$dset" == "mls" ]]; then
    splits="mls_fr_test mls_nl_test mls_pt_test mls_de_test mls_es_test mls_it_test mls_pl_test"
    espnet_task=asr2
    task=asr
    input_format=espnet
    input_path=/ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/${espnet_task}/dump/audio_raw/
elif [[ "$dset" == "CoVoST-2" ]]; then
    splits="test.ca-en test.en-ar test.en-cy test.en-et test.en-id test.en-lv test.en-sl test.en-ta test.en-zh-CN test.fr-en test.zh-CN-en test.de-en test.en-ca test.en-de test.en-fa test.en-ja test.en-mn test.en-sv-SE test.en-tr test.es-en test.ja-en"
    espnet_task=st1
    task=st
    input_format=espnet
    input_path=/ocean/projects/cis210027p/pyf98/OWSM_test/${dset}/
elif [[ "$dset" == "slue-voxceleb" ]]; then
    splits="test "
    espnet_task=slu1
    task=slu-sa
    input_format=espnet
    input_path=/ocean/projects/cis210027p/shared/corpora/slue-perb_test/${dset}/
    instructions=("Can you identify the sentiment of the speech as being \"positive,\" \"neutral,\" or \"negative\"?"
    "Determine the sentiment of the speech as either \"positive,\" \"neutral,\" or \"negative\"."
    "Analyze the sentiment of the speech and classify it as \"positive,\" \"neutral,\" or \"negative\"."
    "Can you identify the sentiment of the speech as being \"negative,\" \"neutral,\" or \"positive\"?"
    "Determine the sentiment of the speech as either \"negative,\" \"neutral,\" or \"positive\"."
    "Analyze the sentiment of the speech and classify it as \"negative,\" \"neutral,\" or \"positive\".")
elif [[ "$dset" == "slue-ted" ]]; then
    splits="test "
    espnet_task=slu1
    task=slu-summ
    input_format=espnet
    input_path=/ocean/projects/cis210027p/shared/corpora/slue-perb_test/${dset}/
    instructions=("Write a short summary for this speech."
    "Briefly summarize this speech."
    "What is a summary of this speech?" # Instruction 1, 2, and 3 are FLAN T-5 prompts
    "Please summarize the speech."
    "Summarize the key points and main arguments presented in the speech."
    "Give a brief summary of the speech.")
elif [[ "$dset" == "slue-sqa-5" ]]; then
    splits="test "
    espnet_task=slu1
    task=slu-sqa
    input_format=espnet
    input_path=/ocean/projects/cis210027p/shared/corpora/slue-perb_test/${dset}/
    opts=(--separators "SEP ANS")
fi



# Generate data files
for split in ${splits}; do     
    output_name="${dset}_${split}"
    if [[ -z "${instructions}" ]]; then
        python3 prep_data.py --input $input_path --split $split --input_format $input_format --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_format}" "${opts[@]}"
        if [[ "$task" == "asr" ]] || [[ "$task" == "st" ]]; then
            python3 prep_data.py --input $input_path --split $split --input_format $input_format --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/specify_src_lang/${output_name}.${output_file_format}" --specify_src_lang "${opts[@]}"
        fi
    else
        for (( i = 0; i < ${#instructions[@]}; i++ )); do
            output_name_i="${output_name}.${i}"
            instruction="${instructions[$i]}"
            python3 prep_data.py --input $input_path --split $split --input_format $input_format --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name_i}.${output_file_format}" --instruction "${instruction}" "${opts[@]}"
            if [[ "$task" == "asr" ]] || [[ "$task" == "st" ]]; then
                python3 prep_data.py --input $input_path --split $split --input_format $input_format --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/specify_src_lang/${output_name_i}.${output_file_format}" --specify_src_lang --instruction "${instruction}" "${opts[@]}"
            fi
        done
    fi
done