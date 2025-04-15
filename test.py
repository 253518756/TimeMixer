import numpy as np
import pandas as pd

file_path = './results/long_term_forecast_ETTh2_96_96_none_TimeMixer_ETTh2_sl96_pl96_dm16_nh8_el2_dl1_df32_fc1_ebtimeF_dtTrue_Exp_0/pred.npy'
data = np.load(file_path)
df = pd.DataFrame(data[0])
# np.savetxt('result.txt', data, delimiter=',', fmt='%s')
print(data.shape)
print(data[0])
csv_path = ("result.csv")
df.to_csv(csv_path, index=False)