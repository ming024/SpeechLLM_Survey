#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
set -o pipefail

dataset_name="librispeech"
task="asr2"
test_sets="test_clean test_other"
tgt_case="ts"
tgt_lang=en

./eval.sh \
    --tgt_case ${tgt_case} \
    --tgt_lang ${tgt_lang} \
    --dataset_name ${dataset_name} \
    --task ${task} \
    --test_sets "${test_sets}" "$@"
