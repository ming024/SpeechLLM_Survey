export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1
FAIRSEQ_ROOT=${PWD}/fairseq
export PYTHONPATH=$$PYTHONPATH:${FAIRSEQ_ROOT}

model_path=$1
subset=$2
data_dir=$3
results_dir=$4

if [ ! -e ${data_dir}/dict.txt ]; then
    ln -s ${PWD}/wavllm/test_data/dict.txt ${data_dir}/dict.txt
fi

src_dir=${model_path%/*}
cpt=${model_path##*/}
cpt=${cpt%.*}

beam_size=1

[ ! -d $results_path ] && mkdir -p $results_path

python $FAIRSEQ_ROOT/examples/wavllm/inference/generate.py $data_dir \
--user-dir examples/wavllm \
--tokenizer-path $FAIRSEQ_ROOT/examples/wavllm/tokenizer/tokenizer.model \
--gen-subset ${subset} \
--task speechllm_task \
--path ${model_path} \
--results-path ${results_dir} \
--max-tokens 1600000 \
--sampling --beam 1 --nbest 1 --temperature 0.5 \
--max-len-a 0 --max-len-b 512