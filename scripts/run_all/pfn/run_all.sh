DSs=("ECL" "ETTh1" "ETTh2" "ETTm1" "ETTm2" "Exchange" "Traffic")
for ds in "${DSs[@]}"; do 
    bash ./scripts/run_all/pfn/"${ds}".sh > ./logs/SimplePFN/"${ds}".log
done