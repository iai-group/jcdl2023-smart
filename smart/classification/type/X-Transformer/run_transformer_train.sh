#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --partition=gpuA100
#SBATCH --time=03:00:00
#SBATCH --job-name=smart_$2_$3_$$4
#SBATCH --output=smart_$2_$3_$$4.out
#echo "before source"
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
#__conda_setup="$('/bhome/vsetty/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
#if [ $? -eq 0 ]; then
#    eval "$__conda_setup"
#else
#    if [ -f "/bhome/vsetty/anaconda3/etc/profile.d/conda.sh" ]; then
#        . "/bhome/vsetty/anaconda3/etc/profile.d/conda.sh"
#    else
#        export PATH="/bhome/vsetty/anaconda3/bin:$PATH"
#    fi
#fi
#unset __conda_setup
# <<< conda initialize <<<
#cd /bhome/vsetty/repos/wsdm2022-smart//smart/classification/type/X-Transformer/

#conda activate xbert_smart
GPID=$1
DATASET=$2
MODEL_TYPE=$3
INDEXER_NAME=$4 # pifa-tfidf-s0 ||| pifa-neural-s0 ||| text-emb-s0
NUM_GPUS=$5
PORT=$6
TEST=$7
PREDICT_ONLY=$8
if [ -z "$TEST" ]
then
    TEST="FALSE"
fi
echo $TEST
OMP_NUM_THREADS=100
if [ ${MODEL_TYPE} == "bert" ]; then
    MODEL_NAME=bert-large-cased-whole-word-masking
elif [ ${MODEL_TYPE} == "roberta" ]; then
    MODEL_NAME=roberta-large
elif [ ${MODEL_TYPE} == "xlnet" ]; then
    MODEL_NAME=xlnet-large-cased
else
    echo "unknown MODEL_TYPE! [ bert | robeta | xlnet ]"
    exit
fi
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
MAX_XSEQ_LEN=256

# Nvidia 2080Ti (11Gb), fp32
PER_DEVICE_TRAIN_BSZ=8
PER_DEVICE_VAL_BSZ=16
GRAD_ACCU_STEPS=4

# Nvidia V100 (16Gb), fp32
PER_DEVICE_TRAIN_BSZ=16
PER_DEVICE_VAL_BSZ=32
GRAD_ACCU_STEPS=2

# set hyper-params by dataset
if [ ${DATASET} == "Eurlex-4K" ]; then
    MAX_STEPS=1000
    WARMUP_STEPS=100
    LOGGING_STEPS=50
    LEARNING_RATE=5e-5
elif [ ${DATASET} == "Wiki10-31K" ]; then
    MAX_STEPS=1400
    WARMUP_STEPS=100
    LOGGING_STEPS=50
    LEARNING_RATE=5e-5
elif [ ${DATASET} == "AmazonCat-13K" ]; then
    MAX_STEPS=20000
    WARMUP_STEPS=2000
    LOGGING_STEPS=100
    LEARNING_RATE=8e-5
elif [ ${DATASET} == "Wiki-500K" ]; then
    MAX_STEPS=80000
    WARMUP_STEPS=1000
    LOGGING_STEPS=100
    LEARNING_RATE=6e-5  # users may need to tune this LEARNING_RATE={2e-5,4e-5,6e-5,8e-5} depending on their CUDA/Pytorch environments
elif [[ ${DATASET} =~ "dbpedia_split" ]]; then
    MAX_STEPS=1400
    WARMUP_STEPS=100
    LOGGING_STEPS=100
    
    LEARNING_RATE=5e-5
elif [[ ${DATASET} =~ "dbpedia_full" ]]; then
    MAX_STEPS=2800
    WARMUP_STEPS=100
    LOGGING_STEPS=100
    if [ ${TEST} == "True" ]; then
        echo "inside test mode"
        MAX_STEPS=7
        WARMUP_STEPS=1
        LOGGING_STEPS=10
    fi
    LEARNING_RATE=6e-5
elif [[ ${DATASET} =~ "dbpedia_summaries_split" ]]; then
    MAX_STEPS=1400
    WARMUP_STEPS=100
    LOGGING_STEPS=100
    LEARNING_RATE=5e-5
elif [[ ${DATASET} =~ "wikidata_split" ]]; then
    MAX_STEPS=1400
    WARMUP_STEPS=100
    LOGGING_STEPS=100
    LEARNING_RATE=5e-5
elif [[ ${DATASET} =~ 'wikidata_full' ]]; then
     MAX_STEPS=4000
     WARMUP_STEPS=200
     LOGGING_STEPS=100
     if [[ ${TEST} == "True" ]]; then
        MAX_STEPS=10
        WARMUP_STEPS=2
        LOGGING_STEPS=10
    fi
     LEARNING_RATE=6e-5
else
    MAX_STEPS=1400
    WARMUP_STEPS=100
    LOGGING_STEPS=100
    LEARNING_RATE=5e-5
fi

MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher/${MODEL_NAME}
mkdir -p ${MODEL_DIR}

if [[ ${PREDICT_ONLY} == "True" ]]; then
    echo "PREDICT_ONLY set to True, skipping training"
fi

# if [[ ${PREDICT_ONLY} == "False" ]]; then
# train
CUDA_VISIBLE_DEVICES=${GPID} python -m torch.distributed.launch \
    --nproc_per_node ${NUM_GPUS} xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} --do_train \
    -x_train ${PROC_DATA_DIR}/X.train.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_train ${PROC_DATA_DIR}/C.train.${INDEXER_NAME}.npz \
    -o ${MODEL_DIR} --overwrite_output_dir \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --logging_steps ${LOGGING_STEPS} \
    --port ${PORT} \
    |& tee ${MODEL_DIR}/log.txt 

CUDA_VISIBLE_DEVICES=${GPID} python -u xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} \
    -x_train ${PROC_DATA_DIR}/X.train.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_train ${PROC_DATA_DIR}/C.train.${INDEXER_NAME}.npz \
    -x_test ${PROC_DATA_DIR}/X.test.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_test ${PROC_DATA_DIR}/C.test.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}

#### end ####

