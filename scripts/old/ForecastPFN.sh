is_training=0
model=ForecastPFN
data_path="../ForecastPFN/academic_data"
# model_path="../ForecastPFN/src/synthetic_generation/data/synthetic/models/ForecastPFN"


for preLen in 6 8 14 18 24 36 48 60
do

# exchange
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data exchange \
 --root_path $data_path/exchange_rate/ \
 --data_path exchange_rate.csv \
 --model $model \
 --model_id Exchange_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi


# illness
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data ili \
 --root_path $data_path/illness/ \
 --data_path national_illness.csv \
 --model $model \
 --model_id ili_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi


# weather
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data weather-mean \
 --root_path $data_path/weather/ \
 --data_path weather_agg.csv \
 --model $model \
 --model_id Weather_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi


# traffic
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data traffic-mean \
 --root_path $data_path/traffic/ \
 --data_path traffic_agg.csv \
 --model $model \
 --model_id Traffic_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi


# electricity
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data ECL-mean \
 --root_path $data_path/electricity/ \
 --data_path electricity_agg.csv \
 --model $model \
 --model_id ECL_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi


# ETTh1
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data ETTh1-mean \
 --root_path $data_path/ETT-small/ \
 --data_path ETTh1_agg.csv \
 --model $model \
 --model_id ETTh1_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi


# ETTh2
python run.py \
 --task_name forecastpfn \
 --is_training $is_training \
 --data ETTh2-mean \
 --root_path $data_path/ETT-small/ \
 --data_path ETTh2_agg.csv \
 --model $model \
 --model_id ETTh2_36_18_$preLen \
 --seq_len 36 \
 --label_len 18 \
 --pred_len $preLen \
 --itr 1 \

nvidia-smi

done;


