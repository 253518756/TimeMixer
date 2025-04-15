import akshare as ak
import pandas as pd

# 获取中证 1000 成分股信息
try:
    # 尝试使用新的参数名或者调用方式
    index_stock_cons_df = ak.index_stock_cons(symbol="000852")
except TypeError:
    print("当前使用的 akshare 版本中 index_stock_cons 函数可能不支持 'index' 或 'symbol' 参数，请检查文档。")
    # 若上述方法不可用，可考虑手动查找其他途径获取成分股信息
    index_stock_cons_df = None

# print(index_stock_cons_df)
# csv_path = ("csi1000_stocks.csv")
# index_stock_cons_df.to_csv(csv_path, index=False)

if index_stock_cons_df is not None:
    all_stocks_data = []

    script = """
export CUDA_VISIBLE_DEVICES=0

model_name=TimeMixer

seq_len=32
pred_len=32
e_layers=2
down_sampling_layers=3
down_sampling_window=2
learning_rate=0.01
d_model=16
d_ff=32
train_epochs=10
patience=10
batch_size=16
target=涨跌幅

    """

    # 遍历每支成分股
    for _, row in index_stock_cons_df.iterrows():
        stock_code = row['品种代码']

        sentence = """
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path  ./dataset/CSI1000/ \
    --data_path """ + stock_code + """ \
    --model_id CSI1000_$seq_len_$pred_len \
    --model $model_name \
    --target $target \
    --data CSI1000 \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --enc_in 10 \
    --c_out 10 \
    --des 'Exp' \
    --itr 1 \
    --d_model $d_model \
    --d_ff $d_ff \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --batch_size 128 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --inverse \
        """
        print(sentence)
        script += sentence + "\n"


    # 保存脚本到文件
    with open("./scripts/TimeMixer_CSI1000_32_32.sh", "w") as f:
        f.write(script)
    print(f"所有股票数据已保存到 ./dataset/./scripts/TimeMixer_CSI1000_96_32.sh!")