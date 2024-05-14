#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
# set -o pipefail

task=slue-summ
test_sets="test "

./eval.sh \
    --test_sets "${test_sets}" "$@"
