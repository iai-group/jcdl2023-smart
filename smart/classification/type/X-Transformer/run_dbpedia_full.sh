#conda activate pt1.2_xmlc_transformer
dataset=$1
echo $dataset
model=$2
echo $model
# emb=pifa-tfidf
emb=$3
gpus=$4
num_gpus=$5
clusters=$6
port=$7
Test=$8
PREDICT_ONLY=$9
sh ./run_preprocess_label.sh $dataset $emb $clusters && \
sh ./run_preprocess_feat.sh $dataset $model 256 && \
#Change the gpu ids you want to use here and num gpus which is the last param
sh ./run_transformer_train.sh $gpus $dataset $model $emb-s0 $num_gpus $port $Test $PREDICT_ONLY && \
sh ./run_transformer_predict.sh $dataset $model $emb-s0
