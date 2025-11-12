is_training=1
model=TSMixer
data_path="../ForecastPFN/academic_data"
# model_path="../ForecastPFN/src/synthetic_generation/data/synthetic/models/ForecastPFN"


for preLen in 6 8 14 18 24 36 48 60
do

# for is_training in 1 0; do

# exchange
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data exchange \
 --root_path $data_path/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --model_id Exchange_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi


# illness
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data ili \
 --root_path $data_path/illness/ \
 --data_path national_illness.csv \
 --model $model \
 --model_id ili_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi


# weather
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data weather-mean \
 --root_path $data_path/weather/ \
 --data_path weather_agg.csv \
 --model $model \
 --model_id Weather_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi


# traffic
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data traffic-mean \
 --root_path $data_path/traffic/ \
 --data_path traffic_agg.csv \
 --model $model \
 --model_id Traffic_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi


# electricity
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data ECL-mean \
 --root_path $data_path/electricity/ \
 --data_path electricity_agg.csv \
 --model $model \
 --model_id ECL_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi


# ETTh1
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data ETTh1-mean \
 --root_path $data_path/ETT-small/ \
 --data_path ETTh1_agg.csv \
 --model $model \
 --model_id ETTh1_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi


# ETTh2
python run.py \
 --task_name long_term_forecast \
 --is_training $is_training \
 --data ETTh2-mean \
 --root_path $data_path/ETT-small/ \
 --data_path ETTh2_agg.csv \
 --model $model \
 --model_id ETTh2_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --e_layers 2 \
 --d_layers 1 \
 --factor 3 \
 --enc_in 1 \
 --dec_in 1 \
 --c_out 1 \
 --itr 5 \
 --n_heads 4 \
 --d_model 1024

nvidia-smi



# done;

for i in 0 1 2 3
do
  dirs=$(find checkpoints -type d -name "*_test_${i}")
  if [ -n "$dirs" ]; then
    echo "Deleting: $dirs"
    rm -rf $dirs
  fi
done
done