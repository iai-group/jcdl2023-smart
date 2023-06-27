#!/bin/bash
# This script prepares the data for xbert and copies it to the corresponding  
# folders. And for each embedding type, and dataset the scripts to train and
# evaluate are run.
eval "$(conda shell.bash hook)"
# conda activate st
echo ${dataset}
# conda deactivate
Test=True
#EMB_LIST=( pifa-tfidf text-emb pifa-neural )
EMB_LIST=( type-features )

conda activate emnlp2021_smart
dataset_list=( dbpedia )
#dataset_list=( wikidata dbpedia )
model_list=( xlnet )
#model_list=( roberta xbert )

for model in "${model_list[@]}"; do	
for dataset in "${dataset_list[@]}"; do
	python -m smart.classification.type.prepare_data_for_xbert ${dataset} ${Test}
		for EMB in "${EMB_LIST[@]}"; do
		echo ${dataset}
		echo ${model}
		echo ${EMB}
 	        echo "./run_dbpedia_full.sh ${dataset}_full ${model} ${EMB} 7 1 ${Test}" 
		cd ./smart/classification/type/X-Transformer/ && pip install -e . && python setup.py install --force &&\
 	       ./run_dbpedia_full.sh ${dataset}_full ${model} ${EMB} 1,3 2 ${Test} && cd -
          echo "running predictions now"
		python -m smart.classification.type.predict_types_using_xbert ${dataset} ${EMB} ${model}
		if [ $dataset = dbpedia ]
		then
            echo "running dbpedia evaluation now"
            python  data/smart_dataset/${dataset}/evaluation/evaluate.py \
            --type_hierarchy_tsv data/smart_dataset/${dataset}/evaluation/${dataset}_types.tsv \
            --ground_truth_json data/smart_dataset/${dataset}/smarttask_${dataset}_test.json  \
            --system_output_json data/runs/${dataset}/${model}_${EMB}_test_pred.json > data/runs/${dataset}_${EMB}_${model}_result.txt
            cat data/runs/${dataset}_${EMB}_${model}_result.txt
		fi
        if [ $dataset = wikidata ]
        then
        echo "running wikidata evaluation now"
            cd data/smart_dataset/wikidata/evaluation/code_files/ && \
            python  wikidata_eval.py \
             --ground_truth_json ../../../../../data/smart_dataset/wikidata/lcquad2_anstype_wikidata_test_gold.json \
            --system_output_json ../../../../runs/${dataset}/${model}_${EMB}_test_pred.json >  ../../../../runs/${dataset}_${EMB}_${model}_result.txt
            cat  ../../../../runs/${dataset}_${EMB}_${model}_result.txt
            cd -
        fi 
		done
	done
done
