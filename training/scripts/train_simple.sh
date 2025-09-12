# for pred_len in 6 8 14 18 24 36 48 60
for pred_len in 36
do
python train.py --seq_len 36 --pred_len $pred_len
done