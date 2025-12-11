#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./queue_from_csv.sh path/to/config.csv [MODEL_NAME]
# Example:
#   ./queue_from_csv.sh ./configs/pfn_grid.csv LinearPFNv2
#
# Notes:
# - Expects a header line exactly like:
#   L,H,d,Lblk,n,dff,do,Cmin,Cmax,Qmin,Qmax,data_version,tested
# - Processes only rows with tested=0
# - After a row finishes successfully, it updates that row's "tested" to 1 in the CSV.

CSV_FILE="${1:-configs.csv}"
MODEL="${2:-LinearPFN}"

# Datasets to iterate
DSs=("ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Exchange" "Traffic" "Weather")

# --- helper: atomically set "tested" to 1 for a given line number ---
mark_tested() {
  local csv="$1"
  local line_no="$2"
  # Update only the last field (tested) on that specific line
  awk -F',' -v OFS=',' -v target="$line_no" '
    NR==target { $NF=1 } { print }
  ' "$csv" > "${csv}.tmp" && mv "${csv}.tmp" "$csv"
}

# --- pull runnable rows via awk (header-aware, tested==0), and include line number (NR) ---
# We output as tab + comma CSV to keep parsing robust:
#   NR \t L,H,d,Lblk,n,dff,do,Cmin,Cmax,Qmin,Qmax,data_version,tested
awk -F',' '
  function trim(s){ gsub(/^[[:space:]]+|[[:space:]]+$/,"",s); return s }
  NR==1 {
    for (i=1;i<=NF;i++) { h[trim($i)]=i }
    # sanity check (optional)
    next
  }
  {
    # ensure we’re working with trimmed fields
    for (i=1;i<=NF;i++) fld[i]=trim($i)
    if (fld[h["tested"]]==0) {
      # Print line number (NR) then the whole row again
      print NR "\t" $0
    }
  }
' "$CSV_FILE" | while IFS=$'\t' read -r ROW_NR ROW_DATA; do
  # Now parse the row’s comma fields into variables
  IFS=',' read -r L H d_model Lblk n dff do Cmin Cmax Qmin Qmax data_version tested <<< "$ROW_DATA"

  # Clean possible whitespace around numeric fields
  L=$(echo "$L" | xargs); H=$(echo "$H" | xargs)
  d_model=$(echo "$d_model" | xargs); Lblk=$(echo "$Lblk" | xargs)
  n=$(echo "$n" | xargs); dff=$(echo "$dff" | xargs)
  do=$(echo "$do" | xargs); data_version=$(echo "$data_version" | xargs)
  Cmin=$(echo "$Cmin" | xargs); Cmax=$(echo "$Cmax" | xargs)

  echo ">>> Running: MODEL=$MODEL L=$L H=$H d=$d_model Lblk=$Lblk n=$n dff=$dff do=$do Cmin=$Cmin Cmax=$Cmax data_version=$data_version (CSV line $ROW_NR)"

  # Log directory consistent with your previous naming
  log_dir=/${MODEL}/v${data_version}/${L}_${H}_d${d_model}_L${Lblk}_n${n}_dff${dff}_do${do}
  mkdir -p "./logs${log_dir}"

  # Run each dataset sequentially
  for ds in "${DSs[@]}"; do
    echo " -> [$ds]"
    bash "./scripts/run_all/pfn/${ds}.sh" \
      --model "$MODEL" \
      --seq_len "$L" \
      --d_model "$d_model" \
      --n_heads "$n" \
      --e_layers "$Lblk" \
      --d_ff "$dff" \
      --data_version "$data_version" \
      --c_min "$Cmin" \
      --c_max "$Cmax" \
      > "./logs${log_dir}/${ds}.log"
  done

  # Post-run summary (kept from your original script)
  python log_tensorboard_test.py \
    --model "$MODEL" \
    --data_version "$data_version" \
    --seq_len "$L" --pred_len "$H"\
    --d_model "$d_model" \
    --n_heads "$n" \
    --e_layers "$Lblk" \
    --d_ff "$dff" \
    --c_min "$Cmin" \
    --c_max "$Cmax" 

  # Mark this row as tested=1 only after all above succeed
  mark_tested "$CSV_FILE" "$ROW_NR"
  echo " ✓ Marked CSV line $ROW_NR as tested=1"
done

echo "All pending (tested=0) rows processed."
