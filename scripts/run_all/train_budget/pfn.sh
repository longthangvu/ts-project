model=SimplePFN
data_path="../ForecastPFN/academic_data"

for train_budget in 0 50 500
do

for pred_len in 36
# for pred_len in 6 8 14 18 24 36 48 60
do

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data ili \
  --root_path $data_path/illness \
  --data_path national_illness.csv \
  --model_id Ili_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data exchange \
  --root_path $data_path/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data weather-mean \
  --root_path $data_path/weather/ \
  --data_path weather_agg.csv \
  --model_id Weather_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data traffic-mean \
  --root_path $data_path/traffic/ \
  --data_path traffic_agg.csv \
  --model_id Traffic_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data ECL-mean \
  --root_path $data_path/electricity/ \
  --data_path electricity_agg.csv \
  --model_id ECL_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget

python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data ETTh1-mean \
  --root_path $data_path/ETT-small/ \
  --data_path ETTh1_agg.csv \
  --model_id ETTh1_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget
  
python run.py \
  --task_name meta_learning_pfn \
  --is_training 0 \
  --data ETTh2-mean \
  --root_path $data_path/ETT-small/ \
  --data_path ETTh2_agg.csv \
  --model_id ETTh2_36_36_$model \
  --model $model \
  --features S \
  --seq_len 36 --label_len 36 \
  --pred_len $pred_len \
  --d_model 512 \
  --n_heads 16 \
  --e_layers 12 \
  --d_ff 2048 \
  --train_budget $train_budget

done;
done