#!/bin/bash
export HF_HOME="./datas" 
export HF_ENDPOINT=https://hf-mirror.com
DATASETS_PATH="hf"
OUTPUT_PATH="./eval_results_medgemma-27b-it"
EVAL_DATASETS="RadGenome_Brain_MRI"

MODEL_NAME="MedGemma"
MODEL_PATH="/path/to/your/ckpt/medgemma-27b-it"


CUDA_VISIBLE_DEVICES="0,1"
USE_VLLM="False"
IFS=',' read -r -a GPULIST <<< "$CUDA_VISIBLE_DEVICES"
TOTAL_GPUS=${#GPULIST[@]}
CHUNKS=$TOTAL_GPUS  
#Eval setting
SEED=42
REASONING="False"
TEST_TIMES=1

# Eval LLM setting
MAX_NEW_TOKENS=40960
MAX_IMAGE_NUM=600
TEMPERATURE=0
TOP_P=0.95
REPETITION_PENALTY=1.0


# LLM judge setting
USE_LLM_JUDGE="False"
GPT_MODEL="gpt-4.1-2025-04-14"
OPENAI_API_KEY="Your key"


for IDX in $(seq 0 $((CHUNKS-1))); do

    echo "Starting chunk $IDX on GPU ${GPULIST[$IDX]}"
    export chunk_idx=$IDX
    export num_chunks=$CHUNKS

    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python eval.py \
    --eval_datasets "$EVAL_DATASETS" \
    --datasets_path "$DATASETS_PATH" \
    --output_path "$OUTPUT_PATH" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --seed $SEED \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_image_num "$MAX_IMAGE_NUM" \
    --use_vllm "$USE_VLLM" \
    --num_chunks $CHUNKS \
    --chunk_idx $IDX \
    --reasoning $REASONING \
    --temperature "$TEMPERATURE"  \
    --top_p "$TOP_P" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --use_llm_judge "$USE_LLM_JUDGE" \
    --judge_gpt_model "$JUDGE_GPT_MODEL" \
    --openai_api_key "$OPENAI_API_KEY" \
    --test_times "$TEST_TIMES"  &  
done


wait

