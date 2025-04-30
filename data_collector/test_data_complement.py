import pandas as pd

# 该脚本的主要功能是从train.csv的数据中补充test.csv股票数据，以满足实现不局限于5天预测1天的任务，例如32天数据预测1天。

# 定义要提取的每个股票代码分组的最后时间点数量
last_n_points = 32

try:
    # 读取 CSV 文件
    df_train = pd.read_csv('./dataset/CSI300/train.csv', dtype={'股票代码': str})
    df_test = pd.read_csv('./dataset/CSI300/test.csv', dtype={'股票代码': str})

    # 合并数据
    df_combined = pd.concat([df_train, df_test], axis=0)

    # 将日期列转换为 datetime 类型,股票代码转换为字符串类型，并按股票代码和日期排序
    df_combined['日期'] = pd.to_datetime(df_combined['日期'])
    df_sorted = df_combined.sort_values(by=['股票代码', '日期'])

    # 提取每个股票代码分组的最后 last_n_points 个时间点
    df_last_n = df_sorted.groupby('股票代码').tail(last_n_points)



    # 重置索引
    df_last_n.reset_index(drop=True, inplace=True)

    # 生成新的 test.csv
    df_last_n.to_csv('./dataset/CSI300/test_complement.csv', index=False)
    print("新的 test_complement.csv 文件已生成。")

except FileNotFoundError:
    print("错误: 未找到指定的 CSV 文件，请检查文件路径。")
except Exception as e:
    print(f"发生未知错误: {e}")