# Public Bike Scheduling Strategy Based on Demand Prediction for Unbalanced Life-value Distribution

在论文中，我们研究了一个具有不平衡寿命值分布自行车系统的全局调度问题，并设计了一种协调自行车寿命均衡分布的调度方法。

## STSAGCN模型

论文自行车站点需求预测代码实现

### Requirements
* Python 3.8.10
* Pytorch 2.0.1+cu118
* Pandas 2.0.1
* Matplotlib 3.7.1
* Numpy 1.24.4
* seaborn 0.12.2

To install all dependencies:

`pip install -r requirements.txt`

### Usage
* 首先要用 `preprocess_data.py` 对 Citi Bike trip data 进行预处理
```
python preprocess_data.py
``` 
* 其次要用 `generate_dateset.py` 对数据进行划分，划分训练集，验证集和测试集，数据保存在 `data/DataBase/` 文件夹下。
``` 
python generate_datasets.py
``` 
* 模型运行
``` 
python main.py
``` 
* 其他

    * `data/rowData/` 文件夹中存放原始 Citi Bike trip data 数据
    * `data/DataBase/` 文件夹下存放生成数据文件
    * `SaveModel/` 文件夹下存放保存的最优模型和最新模型

## Bike Lifevalue Schedule

* 模型运行
``` 
python main.py
``` 
* 其他
    * `utils/Dynamic.py` 是论文动态调整自行车寿命文件
    * `utils/Schedule.py` 是论文根据站点需求调度自行车
    * `data/rowData/` 文件夹中存放原始 Citi Bike trip data 数据
    * `data/DataBase/` 文件夹下存放生成数据文件

