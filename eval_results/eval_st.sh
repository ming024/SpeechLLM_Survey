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

src_case=lc.rm
tgt_case=lc.rm

src_lang=es                # source language abbrev. id (e.g., es)
tgt_lang=en                # target language abbrev. id (e.g., en)

# [Task dependent] Set the datadir name created by local/data.sh
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
hyp_cleaner=none # Text cleaner for hypotheses (may be used with external tokenizers)

log "$0 $*"

for symlink in pyscripts utils scripts local path.sh cmd.sh; do
    if [ ! -e "./${symlink}" ]; then
        ln -s ../../TEMPLATE/st1/${symlink} ./${symlink}
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
    _data="${espnet_root}/egs2/${dataset_name}/${dset}"
    _dir="./${dataset_name}_${dset}"

    # TODO(jiatong): add asr scoring and inference

    _scoredir="${_dir}/score_bleu"
    mkdir -p "${_scoredir}"

    paste \
        <(<"${_data}/text.${tgt_case}.${tgt_lang}" \
            ${python} -m espnet2.bin.tokenize_text  \
                -f 2- --input - --output - \
                --token_type word \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --remove_non_linguistic_symbols true \
                --cleaner "${cleaner}" \
                ) \
        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/ref.trn.org"

    # NOTE(kamo): Don't use cleaner for hyp
    paste \
        <(<"${_dir}/text"  \
                ${python} -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type word \
                    --non_linguistic_symbols "${nlsyms_txt}" \
                    --remove_non_linguistic_symbols true \
                    --cleaner "${hyp_cleaner}" \
                    ) \
        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
            >"${_scoredir}/hyp.trn.org"

    # remove utterance id
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/ref.trn.org" > "${_scoredir}/ref.trn"
    perl -pe 's/\([^\)]+\)$//g;' "${_scoredir}/hyp.trn.org" > "${_scoredir}/hyp.trn"

    # detokenizer
    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/ref.trn" > "${_scoredir}/ref.trn.detok"
    detokenizer.perl -l ${tgt_lang} -q < "${_scoredir}/hyp.trn" > "${_scoredir}/hyp.trn.detok"

    # rotate result files
    if [ ${tgt_case} = "tc" ]; then
        pyscripts/utils/rotate_logfile.py ${_scoredir}/result.tc.txt
    fi
    pyscripts/utils/rotate_logfile.py ${_scoredir}/result.lc.txt

    if [ ${tgt_case} = "tc" ]; then
        echo "Case sensitive BLEU result (single-reference)" > ${_scoredir}/result.tc.txt
        sacrebleu "${_scoredir}/ref.trn.detok" \
                    -i "${_scoredir}/hyp.trn.detok" \
                    -m bleu chrf ter \
                    >> ${_scoredir}/result.tc.txt

        log "Write a case-sensitive BLEU (single-reference) result in ${_scoredir}/result.tc.txt"
    fi

    # detokenize & remove punctuation except apostrophe
    scripts/utils/remove_punctuation.pl < "${_scoredir}/ref.trn.detok" > "${_scoredir}/ref.trn.detok.lc.rm"
    scripts/utils/remove_punctuation.pl < "${_scoredir}/hyp.trn.detok" > "${_scoredir}/hyp.trn.detok.lc.rm"
    echo "Case insensitive BLEU result (single-reference)" > ${_scoredir}/result.lc.txt
    sacrebleu -lc "${_scoredir}/ref.trn.detok.lc.rm" \
                -i "${_scoredir}/hyp.trn.detok.lc.rm" \
                -m bleu chrf ter \
                -l ${src_lang}-${tgt_lang}\
                >> ${_scoredir}/result.lc.txt
    log "Write a case-insensitve BLEU (single-reference) result in ${_scoredir}/result.lc.txt"
done

log "Successfully finished. [elapsed=${SECONDS}s]"