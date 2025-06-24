#!/bin/bash
MODELS=("iTransformer" "Nonstationary_Transformer" "PatchTST" "PAttn")
for model in "${MODELS[@]}"; do
    mkdir -p logs/${model}
    ./scripts/run_all/${model}.sh > logs/${model}/run.log
done