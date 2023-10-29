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
    if row[11] not in bif:
        sif[row[3]][4].append(row[11])
        bif[row[11]] = [row[3], float(row[5]), float(row[6]), row[2], 1, int(driving_time), WeiBull(1)]
    if sif[row[3]][5] == 0:
        return sif, bif
    candidate = []

    for i in range(sif[row[3]][5]):
        temp_time = datetime.strptime(bif[sif[row[3]][4][i]][3], "%Y-%m-%d %H:%M:%S")
        if temp_time <= start_time: # 如果用户使用时间晚于车辆的还车时间，则调度，否则不调度
            candidate.append(i)
    if not candidate:
        return sif, bif

    m = random.choice(range(len(candidate)))
    if m in range(sif[row[3]][5]):
        csv_writer.writerow(row)

        sif[row[3]][2] += 1  # 当前车站使用量+1
        sif[row[3]][5] -= 1  # 当前车站的车辆数量-1
        sif[row[7]][3] += 1  # 到达车站的到达量+1
        sif[row[7]][5] += 1  # 到达车站的车辆总数+1
        sif[row[7]][6] = row[2]  # 更新车站最后使用时间

        bif[sif[row[3]][4][m]][0] = row[7]  # 车辆停靠车站的信息更新
        bif[sif[row[3]][4][m]][1] = float(row[9])  # 经纬度更新
        bif[sif[row[3]][4][m]][2] = float(row[10])
        bif[sif[row[3]][4][m]][3] = row[2]  # 到达时间更新
        bif[sif[row[3]][4][m]][4] += 1  # 使用次数更新
        bif[sif[row[3]][4][m]][5] += int(driving_time)  # 使用时间更新
        bif[sif[row[3]][4][m]][6] = WeiBull(bif[sif[row[3]][4][m]][4])

        b = sif[row[3]][4].pop(m)  # 分配当前车站的第一辆车，然后返回车辆编号b
        # print(sif[row[7]][4])
        sif[row[7]][4].append(b)  # 将使用的车辆添加到到达车站最后位置
    return sif, bif


def MinUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer):
    if row[3] not in sif:
        sif[row[3]] = [float(row[5]), float(row[6]), 0, 0, [], 0, row[1]]
    if row[7] not in sif:
        sif[row[7]] = [float(row[9]), float(row[10]), 0, 0, [], 0, row[2]]
    if row[11] not in bif:
        sif[row[3]][4].append(row[11])
        bif[row[11]] = [row[3], float(row[5]), float(row[6]), row[2], 1, int(driving_time), WeiBull(1)]
    if sif[row[3]][5] == 0:
        return sif, bif
    
    Max_lifevalue = -1 
    m = -1 
    for i in range(sif[row[3]][5]): 
        temp_time = datetime.strptime(bif[sif[row[3]][4][i]][3], "%Y-%m-%d %H:%M:%S")
        if temp_time <= start_time and Max_lifevalue < bif[sif[row[3]][4][i]][6]:
            Max_lifevalue = bif[sif[row[3]][4][i]][4]
            m = i
    if m in range(sif[row[3]][5]):
        csv_writer.writerow(row)

        sif[row[3]][2] += 1  # 当前车站使用量+1
        sif[row[3]][5] -= 1  # 当前车站的车辆数量-1
        sif[row[7]][3] += 1  # 到达车站的到达量+1
        sif[row[7]][5] += 1  # 到达车站的车辆总数+1
        sif[row[7]][6] = row[2]  # 更新车站最后使用时间

        bif[sif[row[3]][4][m]][0] = row[7]  # 车辆停靠车站的信息更新
        bif[sif[row[3]][4][m]][1] = float(row[9])  # 经纬度更新
        bif[sif[row[3]][4][m]][2] = float(row[10])
        bif[sif[row[3]][4][m]][3] = row[2]  # 到达时间更新
        bif[sif[row[3]][4][m]][4] += 1  # 使用次数更新
        bif[sif[row[3]][4][m]][5] += int(driving_time)  # 使用时间更新
        bif[sif[row[3]][4][m]][6] = WeiBull(bif[sif[row[3]][4][m]][4])

        b = sif[row[3]][4].pop(m)  # 分配当前车站的寿命值最高的车，然后返回车辆编号b
        sif[row[7]][4].append(b)  # 将使用的车辆添加到到达车站

    return sif, bif


def Dynamic(readfilename, bif, sif, method="Random"):
    out = open('./data/DataBase/{}BikeLife.csv'.format(method), 'w', newline='')
    csv_writer = csv.writer(out, dialect='excel')
    csv_writer.writerow(['tripduration', 'starttime', 'stoptime', 'start station id', 'start station name', 'start station latitude', 'start station longitude', 'end station id', 'end station name', 'end station latitude', 'end station longitude', 'bikeid', 'usertype', 'birth year', 'gender'])

    with open(readfilename, 'r') as rf_obj:
        reader = csv.reader(rf_obj)
        next(reader)

        Initialtime = datetime.strptime("2013-07-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        for row in reader:
            start_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
            finish_time = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
            driving_time = (finish_time - start_time).seconds / 60
            if driving_time >=120:
                continue

            if start_time.day != Initialtime.day:
                cif = Tp_medoids_main(sif)
                sif, bif = Scheduling(bif, sif, cif, start_time.day)
                print("Method {} {} 完成调度。".format(method, start_time.strftime("%Y-%m-%d")))
                Initialtime = start_time
            if method=="Random":
                sif, bif = RandomUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer)   # 车站先到先服务分配
            else:
                sif, bif = MinUpdateInfo(sif, bif, row, start_time, driving_time, csv_writer)  # 车站最小使用量分配

    return sif, bif


