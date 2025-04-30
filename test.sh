
export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=32
pred_len=1
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=32
target=涨跌幅

python -u data_collector/test_data_complement.py
python -u run.py     --task_name long_term_forecast     --is_training 0     --root_path  ./dataset/CSI300/     --data_path 000000     --model_id CSI1000_$seq_len_$pred_len     --model $model_name     --target $target     --data CSI300     --features M     --seq_len $seq_len     --label_len 0     --pred_len $pred_len     --e_layers $e_layers     --enc_in 10     --c_out 10     --des 'Exp'     --itr 1     --d_model $d_model     --d_ff $d_ff     --learning_rate $learning_rate     --train_epochs $train_epochs     --patience $patience     --batch_size $batch_size     --down_sampling_layers $down_sampling_layers     --down_sampling_method avg     --down_sampling_window $down_sampling_window     --num_workers 10         
