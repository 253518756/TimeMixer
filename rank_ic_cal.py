# Convert the dictionary to a DataFrame
import pandas as pd
import os
from utils.metrics import IC


folder_path = './results'

# Read the CSV file
df_raw = pd.read_csv('./results/rank_ic_raw.csv')

# 根据日期分组
grouped = df_raw.groupby('date')
# 创建一个空的 DataFrame 用于存储结果
result_df = pd.DataFrame()
# 遍历每个组
for date, group in grouped:
    # 获取预测值和真实值
    preds = group['pred'].values
    trues = group['true'].values

    # 计算 IC 和 RIC
    ic, ric = IC(preds, trues)

    # 将结果添加到 DataFrame
    result_df = pd.concat([result_df, pd.DataFrame({'date': [date], 'ic': [ic], 'ric': [ric]})])

# 将结果保存到 CSV 文件
result_df.to_csv(os.path.join(folder_path, 'rank_ic_result.csv'), index=False)

