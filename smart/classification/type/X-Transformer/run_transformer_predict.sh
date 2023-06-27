#!/bin/bash

DATASET=$1
DATA_DIR=./datasets/${DATASET}
MODEL_TYPE=$2
#LABEL_NAME_ARR=( pifa-tfidf-s0 pifa-neural-s0 text-emb-s0 )
#MODEL_NAME_ARR=( bert-large-cased-whole-word-masking roberta-large xlnet-large-cased )
LABEL_NAME_ARR=( $3 )
if [ ${MODEL_TYPE} == "bert" ]; then
    MODEL_NAME_ARR=bert-large-cased-whole-word-masking
elif [ ${MODEL_TYPE} == "roberta" ]; then
    MODEL_NAME_ARR=roberta-large
elif [ ${MODEL_TYPE} == "xlnet" ]; then
    MODEL_NAME_ARR=xlnet-large-cased
else
    echo "unknown MODEL_TYPE! [ bert | robeta | xlnet ]"
    exit
fi
EXP_NAME=${DATASET}.final

PRED_NPZ_PATHS=""
for LABEL_NAME in "${LABEL_NAME_ARR[@]}"; do
    OUTPUT_DIR=save_models/${DATASET}/${LABEL_NAME}
    INDEXER_DIR=${OUTPUT_DIR}/indexer
    for MODEL_NAME in "${MODEL_NAME_ARR[@]}"; do
        MATCHER_DIR=${OUTPUT_DIR}/matcher/${MODEL_NAME}
        RANKER_DIR=${OUTPUT_DIR}/ranker/${MODEL_NAME}
        mkdir -p ${RANKER_DIR}
        
        # train linear ranker
        python -m xbert.ranker train \
            -x1 ${DATA_DIR}/X.train.npz \
            -x2 ${MATCHER_DIR}/train_embeddings.npy \
            -y ${DATA_DIR}/Y.train.npz \
            -z ${MATCHER_DIR}/C_train_pred.npz \
            -c ${INDEXER_DIR}/code.npz \
            -o ${RANKER_DIR} -t 0.01 \
            -f 0 -ns 2 --mode ranker
        
        # predict final label ranking
        PRED_NPZ_PATH=${RANKER_DIR}/test.pred.npz
        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x1 ${DATA_DIR}/X.test.npz \
            -x2 ${MATCHER_DIR}/test_embeddings.npy \
            -y ${DATA_DIR}/Y.test.npz \
            -z ${MATCHER_DIR}/C_test_pred.npz \
            -f 0 -t noop
        
        # append all prediction path
        PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
    done
done

# final eval
EVAL_DIR=results_transformer-large
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y ${DATA_DIR}/Y.test.npz \
    -e -p ${PRED_NPZ_PATHS} \
    |& tee ${EVAL_DIR}/${EXP_NAME}.txt

