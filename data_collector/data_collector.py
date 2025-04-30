import akshare as ak
import pandas as pd

# 获取成分股信息
try:
    # 尝试使用新的参数名或者调用方式
    index_stock_cons_df = ak.index_stock_cons_sina(symbol="000300")
except TypeError:
    print("当前使用的 akshare 版本中 index_stock_cons 函数可能不支持 'index' 或 'symbol' 参数，请检查文档。")
    # 若上述方法不可用，可考虑手动查找其他途径获取成分股信息
    index_stock_cons_df = None


if index_stock_cons_df is not None:

    # all_stocks_data dataframe
    all_stocks_data = pd.DataFrame()


    print(index_stock_cons_df)
    # 输出成分股信息到CSV文件
    csv_path = ("./dataset/SH300/csi300_stocks.csv")
    index_stock_cons_df.to_csv(csv_path, index=False)

    # 遍历每支成分股
    for _, row in index_stock_cons_df.iterrows():
        stock_code = row['code']
        try:
            # 获取股票历史行情数据，截止到 2025 年 4 月 1 日近十年数据
            stock_hist_df_train = ak.stock_zh_a_hist(symbol=stock_code,
                                                    period="daily",
                                                    start_date="20150420",
                                                    end_date="20250420",
                                                    adjust="qfq")

            print(stock_hist_df_train)

            # 把涨跌幅移到最后一列
            stock_hist_df_train = stock_hist_df_train[['股票代码', '日期', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅','涨跌额','换手率', '涨跌幅']]

            # 将stock_hist_df_train加在all_stocks_data后面
            all_stocks_data = pd.concat([all_stocks_data, stock_hist_df_train], ignore_index=True)

        except Exception as e:
            print(f"获取股票 {stock_code} 数据时出错: {e}")

    # 保存为CSV 文件
    #csv_path = "./dataset/SH300/sh300_train_data.csv"
    csv_path = "./dataset/SH300/sh300_train_data.csv"

    # # all_stocks_data 转为 DataFrame
    # all_stocks_data = pd.concat(all_stocks_data, ignore_index=True)
    all_stocks_data.to_csv(csv_path, index=False)

    print(f"所有股票数据已保存到 ./dataset/SH300")
