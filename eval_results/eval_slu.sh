#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

python=python3       # Specify python to execute espnet commands.

dataset_name=    # Name of dataset.
task=            # Task name, slu-xxx

# [Task dependent] Set the datadir name created by local/data.sh
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.

log "$0 $*"

for symlink in pyscripts utils scripts local steps path.sh cmd.sh db.sh; do
    if [ ! -e "./${symlink}" ]; then
        ln -s ../../TEMPLATE/slu1/${symlink} ./${symlink}
    fi
done

# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(scripts/utils/print_args.sh $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh

log "Scoring"

for dset in ${test_sets}; do
    _data="${espnet_root}/egs2/${dataset_name}/$(cut -d'.' -f1 <<<${dset})"
    _dir="./${dataset_name}_${dset}"

    if [ "${task}" = slu-sa ]; then
        for _type in cer wer; do
            _scoredir="${_dir}/score_${_type}"
            mkdir -p "${_scoredir}"

            if [ "${_type}" = wer ]; then
                # Tokenize text to word level
                paste \
                    <(<"${_data}/text" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type word \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"


            elif [ "${_type}" = cer ]; then
                # Tokenize text to char level
                paste \
                    <(<"${_data}/text" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type char \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                --cleaner "${cleaner}" \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text"  \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type char \
                                --non_linguistic_symbols "${nlsyms_txt}" \
                                --remove_non_linguistic_symbols true \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

            elif [ "${_type}" = ter ]; then
                # Tokenize text using BPE
                paste \
                    <(<"${_data}/text" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type bpe \
                                --bpemodel "${bpemodel}" \
                                --cleaner "${cleaner}" \
                            ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/ref.trn"

                # NOTE(kamo): Don't use cleaner for hyp
                paste \
                    <(<"${_dir}/text" \
                            ${python} -m espnet2.bin.tokenize_text  \
                                -f 2- --input - --output - \
                                --token_type bpe \
                                --bpemodel "${bpemodel}" \
                                ) \
                    <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                        >"${_scoredir}/hyp.trn"

            fi

            sclite \
                ${score_opts} \
                -r "${_scoredir}/ref.trn" trn \
                -h "${_scoredir}/hyp.trn" trn \
                -i rm -o all stdout > "${_scoredir}/result.txt"

            log "Write ${_type} result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
        done

        [ -f local/score.sh ] && local/score.sh ${local_score_opts} $PWD "./${dataset_name}_${dset}/" "./${dataset_name}_${dset}/"
    elif [ "${task}" = slu-summ ]; then
        python pyscripts/utils/score_summarization.py "${_data}/text" "${_dir}/text"  1> ${_dir}/summ_score.txt
    elif [ "${task}" = slu-sqa ]; then
        python pyscripts/utils/score_summarization.py "${_dir}/ref_text" "${_dir}/text"  1> ${_dir}/summ_score.txt
        python pyscripts/utils/score_rougel.py "${_dir}/ref_text" "${_dir}/text"  1> ${_dir}/rougel_score.txt
    fi
done

log "Successfully finished. [elapsed=${SECONDS}s]"
