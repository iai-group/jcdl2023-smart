#conda activate pt1.2_xmlc_transformer
dataset=$1
echo $dataset
model=$2
echo $model
# emb=pifa-tfidf
emb=$3
gpus=$4
num_gpus=$5
Test=$6
./run_preprocess_label.sh $dataset $emb && \
./run_preprocess_feat.sh $dataset $model 256 && \
#Change the gpu ids you want to use here and num gpus which is the last param
./run_transformer_train.sh $gpus $dataset $model $emb-s0 $num_gpus $Test && \
./run_transformer_predict.sh $dataset $model $emb-s0