#!/bin/bash
# This script prepares the data for xbert and copies it to the corresponding  
# folders. And for each embedding type, and dataset the scripts to train and
# evaluate are run.
source ~/.bashrc
cd /bhome/vsetty/repos/smart
eval "$(conda shell.bash hook)"

echo ${dataset}
# uenv verbose cuda-11.4 cudnn-11.4-8.2.4
# uenv verbose cuda-10.0 cudnn-10.0-7.6.3
# uenv miniconda-python39
# conda deactivate
conda activate xbert_smart
which python
Test=$1
PREDICT_ONLY=False
# EMB_LIST=( text-emb pifa-neural pifa-tfidf type-features )

EMB_LIST=( text-emb )

# conda activate xbert_smart
ulimit -Sv 12000000
#dataset_list=( trec )
dataset_list=( wikidata )
model_list=( xlnet )
#model_list=( roberta xbert )
# use_clean="--use_clean"
use_clean=""
# oracle="--use_oracle_for_cat_classifier"
oracle=""
# cluster_sizes=( 16 32 128 256 )
cluster_sizes=( 16 )
port=44515
for model in "${model_list[@]}"; do	
for dataset in "${dataset_list[@]}"; do
    
	# python -m smart.classification.type.prepare_data_for_xbert --dataset ${dataset} ${use_clean} --use_entity_descriptions
		for EMB in "${EMB_LIST[@]}"; do
        for cluster_size in "${cluster_sizes[@]}"; do
        echo $dataset $model $EMB
		echo ${dataset}
		echo ${model}
		echo ${EMB}
 	        echo "./run_dbpedia_full.sh ${dataset}_full ${model} ${EMB} 7 1 ${Test}" 
 			cd ./smart/classification/type/X-Transformer/ && pip install -e . && python setup.py install --force > xbert_install.logs && \
 	       sh ./run_dbpedia_full.sh ${dataset}_full ${model} ${EMB} 0,1,2,3 4 ${cluster_size} ${port} ${Test} > ${dataset}_full_${model}_${EMB}.logs && cd -
          echo "running predictions now"
		# python -m smart.classification.type.predict_types_using_xbert --dataset ${dataset} --embedding ${EMB} --model ${model}  --use_oracle_for_cat_classifier
        python -m smart.classification.type.predict_types_using_xbert --dataset ${dataset} --embedding ${EMB} --model ${model} --clusters ${cluster_size} ${use_clean} ${oracle}
		if [ $dataset = dbpedia ];
		then
            echo "running dbpedia evaluation now"
                ${dataset}/evaluation/evaluate.py \
            --type_hierarchy_tsv data/smart_dataset/dbpedia/evaluation/dbpedia_types.tsv \
            --ground_truth_json data/smart_dataset/dbpedia/smarttask_dbpedia_test.json  \
            --system_output_json data/runs/dbpedia/${model}_${EMB}_test_pred.json > data/runs/dbpedia_${EMB}_${model}_result.txt
            cat data/runs/dbpedia_${EMB}_${model}_result.txt
		fi
        if [ $dataset = wikidata ];
        then
        echo "running wikidata evaluation now"
            oracle_option="False"
            use_clean_option="False"
            cd data/smart_dataset/wikidata/evaluation/code_files/ && \
            python  wikidata_eval.py \
             --ground_truth_json ../../../../../data/smart_dataset/wikidata/lcquad2_anstype_wikidata_test_gold.json \
            --system_output_json ../../../../runs/${dataset}/${model}_${EMB}_oracle_${oracle_option}_clean_${use_clean_option}_${cluster_size}_test_pred.json  >  ../../../../runs/${dataset}_${EMB}_${model}_oracle_${oracle_option}_clean_${use_clean_option}_${cluster_size}_result.txt
            cat ../../../../runs/${dataset}_${EMB}_${model}_oracle_${oracle_option}_clean_${use_clean_option}_${cluster_size}_result.txt
            cd -
        fi 
        cp -r smart/classification/type/X-Transformer/save_models/wikidata_full smart/classification/type/X-Transformer/save_models/wikidata_full_${oracle_option}_${use_clean_option}_${cluster_size}
		done
	done
done

done
