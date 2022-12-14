#!/bin/bash

GRAPH_DATASET_PATH=$DATASETS/graph
GRAPH_DATASETS=(gplus_108K_13M_csr_float32.npz
                ogbl_ppa_576K_42M_csr_float32.npz
                hollywood_1M_113M_csr_float32.npz
                pokec_1633K_31M_csr_float32.npz
                ogbn_products_2M_124M_csr_float32.npz
                mouse_gene_45K_29M_csr_float32.npz)

bitstream=$1

mkdir -p ./results
for ((i = 0; i < ${#GRAPH_DATASETS[@]}; i++)) do
    name_parts=(${GRAPH_DATASETS[i]//_/ })
    name=${name_parts[0]}
    for ((j = 1; j < ${#name_parts[@]}-4; j++)) do
        name="${name}_${name_parts[j]}"
    done
    ./benchmark $name $GRAPH_DATASET_PATH/${GRAPH_DATASETS[i]} $bitstream "./results/${name}.json"
done

jq -n '[ inputs ]' results/*.json 2>&1 | tee ./result_$(date +%Y%m%d%H%M%S).json
