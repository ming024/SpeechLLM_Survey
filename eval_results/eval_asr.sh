#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
# set -e
# set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

python=python3   # Specify python to execute espnet commands.

dataset_name=    # Name of dataset.
task=            # Task name, as defined in espnet
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.
nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.

tgt_case="ts"
tgt_lang=en      # target language abbrev. id (e.g., en)

log "$0 $*"

for symlink in utils scripts local path.sh cmd.sh; do
    if [ ! -e "./${symlink}" ]; then
        ln -s ../../TEMPLATE/asr1/${symlink} ./${symlink}
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

_dsets="${test_sets}"
for dset in ${_dsets}; do
    _data="/ocean/projects/cis210027p/cchien1/espnet/egs2/${dataset_name}/${task}/dump"
    if [ -e "${_data}/audio_raw" ]; then
        _data="${_data}/audio_raw"
    elif [ -e "${_data}/extracted" ]; then
        _data="${_data}/extracted"
    elif [ -e "${_data}/raw" ]; then
        _data="${_data}/raw"
    else
        log "Error: data dump dir not found in ${_data} "
    fi
    _data="${_data}/${dset}"
    if  [[ "$dset" == *"$dataset_name"* ]]; then
        _dir="${dset}"
    else
        _dir="${dataset_name}_${dset}"
    fi

    for _tok_type in "char" "word"; do
        _opts="--token_type ${_tok_type} "
        _type="${_tok_type:0:1}er"
        _opts+="--non_linguistic_symbols ${nlsyms_txt} "
        _opts+="--remove_non_linguistic_symbols true "

        _scoredir="${_dir}/score_${_type}"
        mkdir -p "${_scoredir}"

        # Tokenize text to ${_tok_type} level
        paste \
            <(perl -p -e 's/^(\S+) (\*\s*)+/$1 /' "${_data}/text" | \
                ${python} -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type ${_tok_type} \
                    --cleaner "${cleaner}" \
                    ${_opts} \
                    ) \
            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                >"${_scoredir}/ref.trn"

        # NOTE(kamo): Don't use cleaner for hyp
        paste \
            <(perl -p -e 's/^(\S+) (\*\s*)+/$1 /' "${_dir}/text" | \
                ${python} -m espnet2.bin.tokenize_text  \
                    -f 2- --input - --output - \
                    --token_type ${_tok_type} \
                    ${_opts} \
                    ) \
            <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                >"${_scoredir}/hyp.trn"
        
        sclite \
            ${score_opts} \
            -r "${_scoredir}/ref.trn" trn \
            -h "${_scoredir}/hyp.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result.txt"

        log "Write ${_type} result in ${_scoredir}/result.txt"
        grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
    done
done

log "Successfully finished. [elapsed=${SECONDS}s]"