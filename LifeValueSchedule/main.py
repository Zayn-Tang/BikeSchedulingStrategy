import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
from utils.InfoOf_bike_station import getInfoOfBike, getInfoOfStation, frequencyDateDicByStation
from utils.Tp_medoids import Tp_medoids
from utils.Scheduling import Scheduling
from utils.Dynamic import Dynamic


# 调度实验所用的训练数据是纽约花旗自行车数据集
Trainingset = [
    "./data/rowData/2014-05 - Citi Bike trip data.csv",
    "./data/rowData/2014-06 - Citi Bike trip data.csv",
    ]
Testset = [
    "./data/rowData/2014-17 - Citi Bike trip data.csv"
    ]

# # 车辆信息：车辆编号、车辆所属车站、所属车站坐标、停靠时的时间戳、车辆总运行次数、车辆总运行时间、车辆寿命值
# # {"16052": ["284", 40.7390169121, -74.0026376103, "2013-11-28 09:02:04", 352, 4335, 0.6465258610151772],...}
InfoOfBike = {'16052': ['504', 40.73221853, -73.98165557, '2013/7/1  0:10:34', 0, 0, 0.0]}
# # 车站信息：车站号、车站坐标、车站进量、车站出量、车站停靠车辆、当前可调度车辆、当前时间戳
InfoOfStation = {'72': [40.76727216, -73.99392888, 0, 0, [], 0, '2013-07-01 00:00:00']}


for t in Trainingset:
    InfoOfBike = getInfoOfBike(t, InfoOfBike)
    InfoOfStation = getInfoOfStation(t, InfoOfStation, InfoOfBike)
with open("./data/DataBase/InfoOfBike.json", 'w') as json_file:
    json.dump(InfoOfBike, json_file, ensure_ascii=False)
with open("./data/DataBase/InfoOfStation.json", 'w') as json_file:
    json.dump(InfoOfStation, json_file, ensure_ascii=False)


# 统计车站的需求量（每天）
frequencyDateDicByStation(Trainingset)


# 进行K-medoids聚类，目的是分治的思想，把一个大问题分解成好几个小问题解决
with open("./data/DataBase/InfoOfStation.json", 'r') as sif:
    InfoOfStation = json.load(sif)
best_SNO = Tp_medoids(InfoOfStation)
with open("./data/DataBase/KmediodsResult.json", 'w') as json_file:
    json.dump(best_SNO, json_file, ensure_ascii=False)

# 先进行一次调度，进行车站需求量归一化和车辆使用量归一化
# 然后进行最大需求量分配最小使用量的标准进行调度，利用每个区域的车站value方差进行调度前后比较
# 保存调度前后value信息，调度后车站车辆的信息更新
# 只是进行一次调度，不改变车子的使用寿命
with open("./data/DataBase/KmediodsResult.json", 'r') as json_file:
    Kmresult = json.load(json_file)
with open("./data/DataBase/InfoOfStation.json", 'r') as sif:
    InfoOfStation = json.load(sif)
with open("./data/DataBase/InfoOfBike.json", 'r') as bif:
    InfoOfBike = json.load(bif)

sif_new, bif_new = Scheduling(InfoOfBike, InfoOfStation, Kmresult, 1)

with open("./data/DataBase/new_InfoOfStation.json", 'w') as nsif:
    json.dump(sif_new, nsif, ensure_ascii=False)
with open("./data/DataBase/new_InfoOfBike.json", 'w') as nbif:
    json.dump(bif_new, nbif, ensure_ascii=False)


# 1. 生成真实使用数据
with open("./data/DataBase/InfoOfBike.json", 'r') as bif:
    InfoOfBike = json.load(bif)
for t in Testset:
    true_InfoOfBike = getInfoOfBike(t, InfoOfBike)
with open("./data/DataBase/TrueBikeLife.json", 'w') as json_file:
    json.dump(true_InfoOfBike, json_file, ensure_ascii=False)


# 2. 生成 Random 调度数据
with open("./data/DataBase/InfoOfBike.json", 'r') as bif_new_file:
    bif_new = json.load(bif_new_file)
with open("./data/DataBase/InfoOfStation.json", 'r') as sif_new_file:
    sif_new = json.load(sif_new_file)

for t in Testset:
    sum_a = 0
    sif_new, bif_new= Dynamic(t, bif_new, sif_new, method="Random")
    for b in bif_new:
        if bif_new[b][6] != 0:
            sum_a += bif_new[b][4]
    print("Random调度: 这个月", t,"的使用总次数：", sum_a)

with open("./data/DataBase/RandomBikeLife.json", 'w') as n8bif:
    json.dump(bif_new, n8bif, ensure_ascii=False)



# 3. 生成 MinUpdate 调度数据
with open("./data/DataBase/InfoOfBike.json", 'r') as bif_new_file:
    bif_new = json.load(bif_new_file)
with open("./data/DataBase/InfoOfStation.json", 'r') as sif_new_file:
    sif_new = json.load(sif_new_file)

for t in Testset:
    sum_a = 0
    sif_new, bif_new= Dynamic(t, bif_new, sif_new, method="MinUpdate")
    for b in bif_new:
        if bif_new[b][6] != 0:
            sum_a += bif_new[b][4]
    print("MinUpdate调度: 这个月", t,"的使用总次数：", sum_a)

with open("./data/DataBase/MinUpdateBikeLife.json", 'w') as n8bif:
    json.dump(bif_new, n8bif, ensure_ascii=False)



# 原始数据和模拟数据对比图像
with open("./data/DataBase/TrueBikeLife.json", 'r') as bif_new_file:
    true_bikelife = json.load(bif_new_file)
with open("./data/DataBase/RandomBikeLife.json", 'r') as bif_new_file:
    Random_bikelife = json.load(bif_new_file)
with open("./data/DataBase/MinUpdateBikeLife.json", 'r') as bif_new_file:
    MinUpdate_bikelife = json.load(bif_new_file)

value_true = []
for b in true_bikelife:
    if true_bikelife[b][6] <= 0.99:
        value_true.append(true_bikelife[b][6])

value_Random = []
for b in Random_bikelife:
    if Random_bikelife[b][6] <= 0.99:
        value_Random.append(Random_bikelife[b][6])

value_MinUpdate = []
for b in MinUpdate_bikelife:
    if MinUpdate_bikelife[b][6] <=0.99:
        value_MinUpdate.append(MinUpdate_bikelife[b][6])

sns.set_style('darkgrid')
plt.figure(figsize=(12,9), dpi= 80)
sns.kdeplot(value_true, shade=True, label='Actual Life-value Choice')
sns.kdeplot(value_Random, shade=True, label='Random Life-value Choice')
sns.kdeplot(value_MinUpdate, shade=True, label='Max Life-value Choice')
plt.legend(loc="upper left", fontsize=14)
plt.title('Bike Life-value Distribution Results', fontsize=14)

plt.xticks(fontsize=14)
plt.xlabel("Bike Life Time", fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel("Density",fontsize=14)

plt.savefig("./data/DataBase/BikeLifeDistribution.png")
plt.show()
plt.close()


