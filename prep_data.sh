model=qwen_audio
output_file_ext=json

# dset=mls
# task=asr
# splits="mls_fr_test mls_nl_test mls_pt_test mls_de_test mls_es_test mls_it_test mls_pl_test"
# for split in ${splits}; do
#     output_name="${split}"
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/asr2/dump/audio_raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/specify_src_lang/${output_name}.${output_file_ext}" --specify_src_lang
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/asr2/dump/audio_raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}"
# done

# dset=librispeech
# task=asr
# splits="test_clean test_other"
# for split in ${splits}; do        
#     output_name="${dset}_${split}"
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/asr2/dump/audio_raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/specify_src_lang/${output_name}.${output_file_ext}" --specify_src_lang
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/asr2/dump/audio_raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}"
# done

# dset=tedlium2
# task=asr
# splits="test "
# for split in ${splits}; do     
#     output_name="${dset}_${split}"
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/asr1/dump/raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/specify_src_lang/${output_name}.${output_file_ext}" --specify_src_lang
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/espnet/egs2/${dset}/asr1/dump/raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}"
# done

# dset=CoVoST-2
# task=st
# splits="test.ca-en test.en-ar test.en-cy test.en-et test.en-id test.en-lv test.en-sl test.en-ta test.en-zh-CN test.fr-en test.zh-CN-en test.de-en test.en-ca test.en-de test.en-fa test.en-ja test.en-mn test.en-sv-SE test.en-tr test.es-en test.ja-en"
# for split in ${splits}; do        
#     output_name="${dset}_${split}"
#     python3 prep_data.py --input /ocean/projects/cis210027p/shared/corpora/OWSM_test/${dset}/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/specify_src_lang/${output_name}.${output_file_ext}" --specify_src_lang
#     python3 prep_data.py --input /ocean/projects/cis210027p/shared/corpora/OWSM_test/${dset}/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}"
# done

dset=slue-voxceleb
task=slu-sa
split="test"
instructions=("Can you identify the sentiment of the speech as being \"positive,\" \"neutral,\" or \"negative\"?"
"Determine the sentiment of the speech as either \"positive,\" \"neutral,\" or \"negative\"."
"Analyze the sentiment of the speech and classify it as \"positive,\" \"neutral,\" or \"negative\"."
"Can you identify the sentiment of the speech as being \"negative,\" \"neutral,\" or \"positive\"?"
"Determine the sentiment of the speech as either \"negative,\" \"neutral,\" or \"positive\"."
"Analyze the sentiment of the speech and classify it as \"negative,\" \"neutral,\" or \"positive\".")
for (( i = 0; i < ${#instructions[@]}; i++ )); do
    output_name="${dset}_${split}.${i}"
    instruction="${instructions[$i]}"
    python3 prep_data.py --input /ocean/projects/cis210027p/shared/corpora/SLUE-PERB_test/${dset}/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}" --instruction "${instruction}"
done

dset=slue-ted
task=slu-summ
split="test"
instructions=("Write a short summary for this speech."
"Briefly summarize this speech."
"What is a summary of this speech?" # Above are FLAN T-5 prompts
"Please summarize the speech."
"Summarize the key points and main arguments presented in the speech."
"Give a brief summary of the speech.")
for (( i = 0; i < ${#instructions[@]}; i++ )); do
    output_name="${dset}_${split}.${i}"
    instruction="${instructions[$i]}"
    python3 prep_data.py --input /ocean/projects/cis210027p/shared/corpora/SLUE-PERB_test/${dset}/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}" --instruction "${instruction}"
done

dset=slue-sqa-5
task=slu-sqa
splits="test "
for split in ${splits}; do
    output_name="${dset}_${split}"
    python3 prep_data.py --input /ocean/projects/cis210027p/shared/corpora/SLUE-PERB_test/${dset}/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}" --separators "SEP ANS"
done

# dset=slue-hvb
# task=slu-dac
# split="test"
# instructions=("Please select one or multiple dialogue acts from the list below that best match the function of the speech: \"question_check\", \"question_repeat\", \"question_general\", \"answer_agree\", \"answer_dis\", \"answer_general\", \"apology\", \"thanks\", \"acknowledge\", \"statement_open\", \"statement_close\", \"statement_problem\", \"statement_instruct\", \"statement_general\", \"backchannel\", \"disfluency\", \"self\", \"other\". Please choose based on the provided options."
# "Select one or more dialogue acts from the options provided below that best describe the purpose of the speech: \"question_check\", \"question_repeat\", \"question_general\", \"answer_agree\", \"answer_dis\", \"answer_general\", \"apology\", \"thanks\", \"acknowledge\", \"statement_open\", \"statement_close\", \"statement_problem\", \"statement_instruct\", \"statement_general\", \"backchannel\", \"disfluency\", \"self\", \"other\"."
# "Choose one or more dialogue acts from the list below that most accurately align with the purpose of the speech: \"question_check\", \"question_repeat\", \"question_general\", \"answer_agree\", \"answer_dis\", \"answer_general\", \"apology\", \"thanks\", \"acknowledge\", \"statement_open\", \"statement_close\", \"statement_problem\", \"statement_instruct\", \"statement_general\", \"backchannel\", \"disfluency\", \"self\", \"other\". Please make your selections based on the provided options.")
# instruction="Please select one or multiple dialogue acts from the list below that best match the function of the speech: \{\
# \"question_check\": \"questions that check/verify information unique to a listener\",\
# \"question_repeat\": \"requests for someone to repeat what they said in order to clarify/understand\",\
# \"question_general\": \"all other questions\",\
# \"answer_agree\": \"Answers indicating a positive response or acceptance\",\
# \"answer_dis\": \"answers indicating a negative response or denial\",\
# \"answer_general\": \"all other answers\",\
# \"apology\": \"a number of often-templated utterances indicating a speaker is apologetic\",\
# \"thanks\": \"a number of often-templated utterances indicating a speaker is appreciative\",\
# \"acknowledge\": \"a response indicating that a speaker has heard, or is empathizing with, what another speaker has said\",\
# \"statement_open\": \"formulaic opening statements that might contain a greeting, introduction, or some other pleasantries\",\
# \"statement_close\": \"formulaic closing statements indicating that the conversation is coming to an end, often containing salutations\",\
# \"statement_problem\": \"an utterance that contains a user\'s primary reason for calling in (this may include questions if the question clearly indicates the call reason)\",\
# \"statement_instruct\": \"an imperative utterance that indicates the speaker wants the listener to do something\",\
# \"statement_general\": \"all other statements\",\
# \"backchannel\": \"verbal or non-verbal expressions indicating the listener\'s attention, agreement, or understanding, while not having much significant meaning on their own\",\
# \"disfluency\": \"filler, reparandum, interregnum\",\
# \"self\": \"essentially rhetorical utterances, or utterances where a speaker is not expecting a response from the listener (i.e. talking to one\'s self)\",\
# \"other\": \"any utterances that don’t fit in any of the above categories, including noise, gibberish, or otherwise uninterpretable speech\",\
# \} Please choose based on the provided options."
# for (( i = 0; i < ${#instructions[@]}; i++ )); do
#     output_name="${dset}_${split}.${i}"
#     instruction="${instructions[$i]}"
#     python3 prep_data.py --input /ocean/projects/cis210027p/cchien1/projects/SLUE-PERB/espnet/egs2/${dset}/slu1_superb/dump/raw/${split}/ --task $task --model $model --dataset $dset --output_json "./eval_data/${model}/not_specify_src_lang/${output_name}.${output_file_ext}" --instruction "${instruction}"
# done