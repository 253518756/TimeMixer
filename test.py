
import numpy as np
from utils.metrics import mCORR
# 示例数据
pred = np.array([1, -2, 3, -4])
true = np.array([2, -3, 4, -5])
# 调用自定义正确率函数
acc = mCORR(pred, true)
print("自定义正确率：", acc)