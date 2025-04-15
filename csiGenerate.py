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

    print(index_stock_cons_df)

    # 遍历每支成分股
    for _, row in index_stock_cons_df.iterrows():
        stock_code = row['品种代码']
        try:
            # 获取股票历史行情数据，截止到 2025 年 4 月 1 日近十年数据
            stock_zh_a_hist_df_train = ak.stock_zh_a_hist(symbol=stock_code,
                                                    period="daily",
                                                    start_date="20150401",
                                                    end_date="20250401",
                                                    adjust="qfq")

            print(stock_zh_a_hist_df_train)

            # 时间戳换名字为date
            stock_zh_a_hist_df_train.rename(columns={'日期': 'date'}, inplace=True)

            # 删除股票代码
            stock_zh_a_hist_df_train.drop(columns=['股票代码'], inplace=True)

            # 把涨跌幅移到最后一列
            stock_zh_a_hist_df_train = stock_zh_a_hist_df_train[['date', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅','涨跌额','换手率', '涨跌幅']]

            # 保存为CSV 文件
            csv_path = "./dataset/CSI1000/csi1000_" + stock_code + "_stocks_data.csv"
            stock_zh_a_hist_df_train.to_csv(csv_path, index=False)
            #
            # stock_zh_a_hist_df_test = ak.stock_zh_a_hist(symbol=stock_code,
            #                                               period="daily",
            #                                               start_date="20230401",
            #                                               end_date="20250401",
            #                                               adjust="qfq")
            #
            # # 时间戳换名字为date
            # stock_zh_a_hist_df_test.rename(columns={'日期': 'date'}, inplace=True)
            #
            # # 删除股票代码
            # stock_zh_a_hist_df_test.drop(columns=['股票代码'], inplace=True)
            #
            # # 把涨跌幅移到最后一列
            # stock_zh_a_hist_df_test = stock_zh_a_hist_df_test[
            #     ['date', '开盘', '收盘', '最高', '最低', '成交量', '成交额', '振幅','涨跌额','换手率','涨跌幅']]
            #
            # # 保存为CSV 文件
            # csv_path = "./dataset/CSI1000/test/csi1000_test_" + stock_code + "_stocks_data.csv"
            # stock_zh_a_hist_df_test.to_csv(csv_path, index=False)
        except Exception as e:
            print(f"获取股票 {stock_code} 数据时出错: {e}")


    print(f"所有股票数据已保存到 ./dataset/CSI1000!")