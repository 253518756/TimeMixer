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

    # 遍历每支成分股
    for _, row in index_stock_cons_df.iterrows():
        stock_code = row['code']
        try:
            # 获取股票历史行情数据，截止到 2025 年 4 月 1 日近十年数据
            stock_hist_df_train = ak.stock_zh_a_hist(symbol=stock_code,
                                                    period="daily",
                                                    start_date="20250428",
                                                    end_date="20250428",
                                                    adjust="qfq")

            print(stock_hist_df_train)

            # 把涨跌幅移到最后一列
            stock_hist_df_train = stock_hist_df_train[['股票代码','涨跌幅']]



            # 将stock_hist_df_train加在all_stocks_data后面
            all_stocks_data = pd.concat([all_stocks_data, stock_hist_df_train], ignore_index=True)

        except Exception as e:
            print(f"获取股票 {stock_code} 数据时出错: {e}")


    # 根据pred排序，获取前十和后十
    path = "../dataset/CSI300/check.csv"
    top_10_max_target = all_stocks_data.sort_values(by='涨跌幅', ascending=False).head(10).drop(['涨跌幅'], axis=1)
    top_10_min_target = all_stocks_data.sort_values(by='涨跌幅', ascending=False).tail(10).drop(['涨跌幅'], axis=1)

    # 重置索引
    top_10_max_target = top_10_max_target.reset_index(drop=True)
    top_10_min_target = top_10_min_target.reset_index(drop=True)

    # 合并两个 DataFrame
    merged_df = pd.concat([top_10_max_target, top_10_min_target], axis=1)

    # 重命名列
    merged_df.columns = ['涨幅最大股票代码', '涨幅最小股票代码']

    merged_df.to_csv(path, index=False)


    print(f"所有股票数据已保存到 ./dataset/SH300")
