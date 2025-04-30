## Baseline说明

文件夹顺序如下

```
|--project
    |--dataset/CSI300（储存数据）
        |--test.csv
        |--train.csv
    |--checkpoint（储存训练模型参数）
    |--run.py（运行代码入口）
    |--train.sh
    |--test.sh
    |--check.csv（储存预测结果）
```

在测评时通过train.sh进行训练，test.sh进行推理，该代码提供给选手参考，选手可以根据自己的需求进行修改，该工程基于TimeMixer实现
注意，提交时需要提交训练的模型结果

<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> (ICLR'24) TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting </b></h2>
</div>

<div align="center">

**[<a href="https://openreview.net/pdf?id=7oLshfEIC2">Paper Page</a>]**
**[<a href="https://iclr.cc/virtual/2024/poster/19347">ICLR Video</a>]**
**[<a href="https://medium.com/towards-data-science/timemixer-exploring-the-latest-model-in-time-series-forecasting-056d9c883f46">Medium Blog</a>]**

**[<a href="https://mp.weixin.qq.com/s/d7fEnEpnyW5T8BN08XRi7g">中文解读1</a>]**
**[<a href="https://mp.weixin.qq.com/s/MsJmWfXuqh_pTYlwve6O3Q">中文解读2</a>]**
**[<a href="https://zhuanlan.zhihu.com/p/686772622">中文解读3</a>]**
**[<a href="https://mp.weixin.qq.com/s/YZ7L1hImIt-jbRT2tizyQw">中文解读4</a>]**

</div>