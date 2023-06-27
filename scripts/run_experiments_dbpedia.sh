#!/bin/bash
# This script prepares the data for xbert and copies it to the corresponding  
# folders. And for each embedding type, and dataset the scripts to train and
# evaluate are run.
eval "$(conda shell.bash hook)"
# conda activate st
echo ${dataset}

# conda deactivate
Test=$1
PREDICT_ONLY=False
# EMB_LIST=( text-emb pifa-neural pifa-tfidf type-features )
# EMB_LIST=( pifa-tfidf type-features text-emb pifa-neural pifa-neural-rdf2vec )
EMB_LIST=( pifa-neural-rdf2vec pifa-tfidf type-feature text-emb pifa-neural )
# EMB_LIST=( text-emb )

conda activate xbert_smart
#dataset_list=( trec )
dataset=dbpedia
model_list=( xlnet roberta )
#model_list=( roberta xbert )p
use_clean="--use_clean"
# use_clean=""
# oracle="--use_oracle_for_cat_classifier"
oracle=""
cluster_sizes=( 64 )
port=44514
# for dataset in "${dataset_list[@]}"; do
for model in "${model_list[@]}"; do	  
	echo "python -m smart.classification.type.prepare_data_for_xbert --dataset ${dataset} ${use_clean}"
	python -m smart.classification.type.prepare_data_for_xbert --dataset ${dataset} ${use_clean} --use_entity_descriptions
		for EMB in "${EMB_LIST[@]}"; do
        for cluster_size in "${cluster_sizes[@]}"; do
        echo $model $dataset $cluster_size $EMB
        echo $dataset $model $EMB
		echo ${dataset}
		echo ${model}
		echo ${EMB}
        echo "running training now"
 	    echo "sh ./run_dbpedia_full.sh ${dataset}_full ${model} ${EMB} 3,7 2 ${cluster_size} ${Test} > ${dataset}_full_${model}_${EMB}.logs"     
 		cd ./smart/classification/type/X-Transformer/ && pip install -e . && python setup.py install --force > xbert_install.logs &&\
 	    sh ./run_dbpedia_full.sh ${dataset}_full ${model} ${EMB} 1,2 2 ${cluster_size} ${port} ${Test} && cd -
        echo "running predictions now"
        echo "python -m smart.classification.type.predict_types_using_xbert --dataset ${dataset} --embedding ${EMB} --model ${model} --clusters ${cluster_size} ${use_clean} ${oracle}"
        python -m smart.classification.type.predict_types_using_xbert --dataset ${dataset} --embedding ${EMB} --model ${model} --clusters ${cluster_size} ${use_clean} ${oracle}
		if [ $dataset = dbpedia ];
		then
            oracle_option="False"
            if [ "$oracle" = "--use_oracle_for_cat_classifier" ];
            then
                oracle_option="True"
            fi 
            use_clean_option="False"
            if [ "$use_clean" = "--use_clean" ];
            then
                use_clean_option="1"
            fi 
            echo "running dbpedia evaluation now"
            python  data/smart_dataset/dbpedia/evaluation/evaluate.py \
            --type_hierarchy_tsv data/smart_dataset/dbpedia/evaluation/dbpedia_types.tsv \
            --ground_truth_json data/smart_dataset/dbpedia/smarttask_dbpedia_test.json  \
            --system_output_json data/runs/dbpedia/${model}_${EMB}_oracle_${oracle_option}_clean_${use_clean_option}_${cluster_size}_test_pred.json > data/runs/dbpedia_${EMB}_${model}_oracle_${oracle_option}_clean_${use_clean_option}_${cluster_size}_result.txt
            cat data/runs/dbpedia_${EMB}_${model}_oracle_${oracle_option}_clean_${use_clean_option}_${cluster_size}_result.txt
		fi
        if [ $dataset = wikidata ];
        then
        echo "running wikidata evaluation now"
            cd data/smart_dataset/wikidata/evaluation/code_files/ && \
            python  wikidata_eval.py \
             --ground_truth_json ../../../../../data/smart_dataset/wikidata/lcquad2_anstype_wikidata_test_gold.json \
            --system_output_json ../../../../runs/${dataset}/${model}_${EMB}_${oracle_option}_${use_clean_option}_${cluster_size}_test_pred.json  >  ../../../../runs/{dataset}_${EMB}_${model}_${oracle_option}_${use_clean_option}_${cluster_size}_result.txt
            cat  ../../../../runs/{dataset}_${EMB}_${model}_${oracle_option}_${use_clean_option}_${cluster_size}_result.txt
            cd -
        fi 
        cp -r smart/classification/type/X-Transformer/save_models/dbpedia_full smart/classification/type/X-Transformer/save_models/dbpedia_full_oracle_${oracle_option}_clean_${use_clean_option}_clusters_${cluster_size}
		done
	done
done
