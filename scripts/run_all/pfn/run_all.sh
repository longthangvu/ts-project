#!/usr/bin/env bash
DSs=("ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Exchange" "Traffic" "Weather")
for f in ./training/ckpts/meta/*; do
    base=$(basename "$f" .json)
    if [[ $base =~ ^([^_]+)_L([0-9]+)_H([0-9]+)_d([0-9]+)_Lblk([0-9]+)_n([0-9]+)_dff([0-9]+)_do([0-9.]+)_v([0-9.]+)$ ]]; then
        model=${BASH_REMATCH[1]}
        L=${BASH_REMATCH[2]}
        H=${BASH_REMATCH[3]}
        d_model=${BASH_REMATCH[4]}
        e_layers=${BASH_REMATCH[5]}
        n_heads=${BASH_REMATCH[6]}
        d_ff=${BASH_REMATCH[7]}
        dropout=${BASH_REMATCH[8]}
        version=${BASH_REMATCH[9]}
        # printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$f" "$model" "$L" "$H" "$d" "$Lblk" "$n" "$dff" "$dropout"
    else
        echo "WARN: skip (name doesn't match): $f" >&2
    fi
    echo $base
    log_dir=/${model}/v${version}/${L}_${H}_d${d_model}_L${e_layers}_n${n_heads}_dff${d_ff}_do${dropout}
    mkdir -p ./logs$log_dir
    for ds in "${DSs[@]}"; do 
        bash ./scripts/run_all/pfn/"${ds}".sh \
            --model $model --seq_len $L \
            --d_model $d_model --n_heads $n_heads \
            --e_layers $e_layers --d_ff $d_ff --data_version $version \
            > ./logs$log_dir/"${ds}".log
    done
    python log_tensorboard_test.py --model $model --data_version $version \
        --seq_len $L --pred_len $H \
        --d_model $d_model --n_heads $n_heads --e_layers $e_layers --d_ff $d_ff
done
