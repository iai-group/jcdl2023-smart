#!/bin/bash

DATASET=$1
LABEL_EMB=$2    # pifa-tfidf | pifa-neural | text-emb
num_clusters=$3

KDIM=50
MAX_ITER=100

cluster_size=10

if [ ${num_clusters} == 64 ]; then
    cluster_size=10
elif [ ${num_clusters} == 32 ]; then
    cluster_size=20
elif [ ${num_clusters} == 16 ]; then
    cluster_size=40
elif [ ${num_clusters} == 128 ]; then
    cluster_size=5
elif [ ${num_clusters} == 256 ]; then
    cluster_size=3
fi

echo "num_clusters" ${num_clusters}
echo "cluster_size" ${cluster_size}
# setup label embedding feature path
# overwrite it if necessary
DATA_DIR=datasets
if [ ${LABEL_EMB} == 'pifa-tfidf' ]; then
    label_emb_inst_path=${DATA_DIR}/${DATASET}/X.train.npz
elif [ ${LABEL_EMB} == 'pifa-neural' ]; then
    label_emb_inst_path=${DATA_DIR}/${DATASET}/X.train.finetune.xlnet.npy
    echo "checking if " $label_emb_inst_path " exists"
   PRETRAINED_FOLDER=save_models/${DATASET}/text-emb-s0/matcher/xlnet-large-cased/train_embeddings.npy
   echo $PRETRAINED_FOLDER
    if test -f "$PRETRAINED_FOLDER"; then
        echo "copying" $label_emb_inst_path
        cp $PRETRAINED_FOLDER ${label_emb_inst_path}
     fi
elif [ ${LABEL_EMB} == 'pifa-neural-rdf2vec' ]; then
    label_emb_inst_path=${DATA_DIR}/${DATASET}/X.train.finetune.rdf2vec.npy
elif [ ${LABEL_EMB} == 'text-emb' ]; then
    label_emb_inst_path=${DATA_DIR}/${DATASET}/X.train.npz
elif [ ${LABEL_EMB} == 'type-features' ]; then
    label_emb_inst_path=${DATA_DIR}/${DATASET}/X.train.npz
fi


# construct label embedding
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}

if [ ${LABEL_EMB} != 'type-features' ]; then
    python -u -m xbert.preprocess \
        --'do_label_embedding' \
        --'do_proc_label' \
        --'do_proc_feat' \
        -i ${DATA_DIR}/${DATASET} \
        -o ${PROC_DATA_DIR} \
        -l ${LABEL_EMB} \
        -x ${label_emb_inst_path}
fi


# semantic label indexing
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
	INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
	python -u -m xbert.indexer \
		-i ${PROC_DATA_DIR}/L.${LABEL_EMB}.npz \
		-o ${INDEXER_DIR} --seed ${SEED} --kdim ${KDIM} \
        --max-iter ${MAX_ITER} --cluster_size ${cluster_size}
done

echo ${DATA_DIR}/${DATASET}
# construct C.[train|test].[label-emb].npz for training matcher
SEED=0
LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
python -u -m xbert.preprocess \
    --do_proc_label \
    -i ${DATA_DIR}/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${INDEXER_DIR}/code.npz

#### end ####

