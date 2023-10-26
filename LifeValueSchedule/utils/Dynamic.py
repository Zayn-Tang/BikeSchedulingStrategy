import csv
import json
import numpy as np
from datetime import datetime
import sys
import  random
from utils.InfoOf_bike_station import WeiBull
from utils.Tp_medoids import Tp_medoids_main
from utils.Scheduling import Scheduling

def RandomUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer):
    if row[3] not in sif:
        sif[row[3]] = [float(row[5]), float(row[6]), 1, 0, [], 0, row[1]]
    if row[7] not in sif:
        sif[row[7]] = [float(row[9]), float(row[10]), 0, 1, [], 0, row[2]]
    if row[11] not in bif:  # 自行车编号
        sif[row[3]][4].append(row[11])
        bif[row[11]] = [row[3], float(row[5]), float(row[6]), row[2], 1, int(driving_time), WeiBull(1)]
    
    candidates = []
    for i in range(sif[row[3]][5]):  # 按队列顺序检索车站的车辆，如果用户使用时间晚于车辆的还车时间，则调度，否则不调度
        temp_time = datetime.strptime(bif[sif[row[3]][4][i]][3], "%Y-%m-%d %H:%M:%S")
        if temp_time <= start_time:
            candidates.append(sif[row[3]][4][i])
    if not candidates:
        return sif, bif

    candidate = random.choice(candidates)
    csv_writer.writerow(row)

    # 更新车站信息
    sif[row[3]][2] += 1  # 出发车站吐量增一
    sif[row[3]][5] -= 1  # 出发车站车子数量减一
    sif[row[3]][4].remove(candidate)  # 出发车站删除候选车子

    sif[row[7]][3] += 1  # 到达车站吞量增一
    sif[row[7]][5] += 1  # 到达车站车子数量增一
    sif[row[7]][4].append(candidate)  # 到达车站增加候选车子
    sif[row[7]][6] = row[2]  # 车子到达时间更新

    # 更新车子信息
    bif[candidate][0] = row[7]  # 车辆停靠车站的信息更新
    bif[candidate][1] = float(row[9])  # 经纬度更新
    bif[candidate][2] = float(row[10])
    bif[candidate][3] = row[2]  # 到达时间更新
    bif[candidate][4] += 1  # 使用次数更新
    bif[candidate][5] += int(driving_time)  # 使用时间更新
    bif[candidate][6] = WeiBull(bif[candidate][4])

    return sif, bif


def MinUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer):
    if row[3] not in sif:
        sif[row[3]] = [float(row[5]), float(row[6]), 1, 0, [], 0, row[1]]
    if row[7] not in sif:
        sif[row[7]] = [float(row[9]), float(row[10]), 0, 1, [], 0, row[2]]
    if row[11] not in bif:  # 自行车编号
        sif[row[3]][4].append(row[11])
        bif[row[11]] = [row[3], float(row[5]), float(row[6]), row[2], 1, int(driving_time), WeiBull(1)]
    
    candidate = -1
    max_bike_value = 0
    noise = 0.8
    for i in range(sif[row[3]][5]):  # 按队列顺序检索车站的车辆，如果用户使用时间晚于车辆的还车时间，则调度，否则不调度
        temp_time = datetime.strptime(bif[sif[row[3]][4][i]][3], "%Y-%m-%d %H:%M:%S")
        if temp_time <= start_time and bif[sif[row[3]][4][i]][6] > max_bike_value and np.random.random() < noise:
            max_bike_value =  bif[sif[row[3]][4][i]][6]
            candidate = i
    if candidate == -1:
        return sif, bif

    candidate = sif[row[3]][4][candidate]

    csv_writer.writerow(row)
    # 更新车站信息
    sif[row[3]][2] += 1  # 出发车站吐量增一
    sif[row[3]][5] -= 1  # 出发车站车子数量减一
    sif[row[3]][4].remove(candidate)  # 出发车站删除候选车子

    sif[row[7]][3] += 1  # 到达车站吞量增一
    sif[row[7]][5] += 1  # 到达车站车子数量增一
    sif[row[7]][4].append(candidate)  # 到达车站增加候选车子
    sif[row[7]][6] = row[2]  # 车子到达时间更新

    # 更新车子信息
    bif[candidate][0] = row[7]  # 车辆停靠车站的信息更新
    bif[candidate][1] = float(row[9])  # 经纬度更新
    bif[candidate][2] = float(row[10])
    bif[candidate][3] = row[2]  # 到达时间更新
    bif[candidate][4] += 1  # 使用次数更新
    bif[candidate][5] += int(driving_time)  # 使用时间更新
    bif[candidate][6] = WeiBull(bif[candidate][4])

    return sif, bif


def Dynamic(readfilename, bif, sif, method="MinUpdate"):
    # Random_MinUpdate 为 True 的时候，采用random的分配方法，如果为False，则使用MinUpdate的方法
    # 假设每次用户骑车的时候，给用户分配该车站中寿命最高的自行车
    # bif 车辆编号：【车站编号，经度坐标，纬度坐标，时间，车辆总运行次数，车辆总运行时间，寿命】
    # sif 车站编号：【车站经度坐标，车站纬度坐标，车站进量，车站出量，[车辆编号]，当前可调度车辆，当前时间戳】

    out = open('./data/DataBase/{}BikeLife.csv'.format(method), 'w', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    csv_writer.writerow(['tripduration', 'starttime', 'stoptime', 'start station id', 'start station name', 'start station latitude', 'start station longitude', 'end station id', 'end station name', 'end station latitude', 'end station longitude', 'bikeid', 'usertype', 'birth year', 'gender'])

    with open(readfilename, "r") as fp:
        reader = csv.reader(fp)
        next(reader)

        PreDay = datetime.strptime("2013-07-01 00:00:00", "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
        for row in reader:
            start_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
            finish_time = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
            driving_time = (finish_time - start_time).seconds / 60
            if driving_time > 180: # 骑行时间过长，删除数据
                continue

            if start_time.hour < 7:
                sif, bif = RandomUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer)
            else:
                # 调度
                if start_time.strftime("%Y-%m-%d") != PreDay:
                    PreDay = start_time.strftime("%Y-%m-%d")
                    cif = Tp_medoids_main(sif)
                    sif, bif = Scheduling(bif, sif, cif, start_time.day)
                    print("Method {} {} 完成调度。".format(method, PreDay))
                if method=="Random":
                    sif, bif = RandomUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer)   # 车站先到先服务分配
                else:
                    sif, bif = MinUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer)  # 车站最小使用量分配
    out.close()
    return sif, bif
