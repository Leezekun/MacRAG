#!/bin/bash

cd ../src

# Define parameters
DATASETS=("hotpotqa" "2wikimultihopqa" "musique")
RAW_CHUNK_SIZE=1500
RAW_OVERLAP=500
SUM_CHUNK_SIZE=450
SUM_OVERLAP=300
EMB_MODELS=("intfloat/multilingual-e5-large" "BAAI/bge-m3")

# Specify CUDA devices at the beginning
export CUDA_VISIBLE_DEVICES="0"

# Step 1: Generate Raw Chunks
echo "Step 1: Generating raw chunks..."
for dataset in "${DATASETS[@]}"; do
    python -W "ignore" gen_index_macrag.py \
        --dataset $dataset \
        --document_chunk_path "../data/corpus/raw/${dataset}.json" \
        --chunk_size $RAW_CHUNK_SIZE \
        --overlap_chunk_size $RAW_OVERLAP
done

# Step 2: Generate Summaries
echo "Step 2: Generating summaries..."
for dataset in "${DATASETS[@]}"; do
    python -W "ignore" gen_index_macrag.py \
        --summary_model "gpt-4o" \
        --chunk_dir "../data/raw_data/${RAW_CHUNK_SIZE}_${RAW_OVERLAP}/${dataset}"
done

# Step 3: Summary Chunk Slicing
echo "Step 3: Summary chunk slicing..."
for emb_model in "${EMB_MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        python -W "ignore" gen_index_macrag.py \
            --chunk_dir "../data/raw_data/${RAW_CHUNK_SIZE}_${RAW_OVERLAP}/${dataset}" \
            --chunk_size $SUM_CHUNK_SIZE \
            --overlap_chunk_size $SUM_OVERLAP \
            --emb_model $emb_model \
            --dataset $dataset &
        
        # Add slight delay between job starts
        sleep 2
    done
    wait
done

echo "Processing complete!"