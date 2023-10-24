import csv
from datetime import datetime
import math
import json

k = - math.pow(10, -8)

def WeiBull(x):
    # x越大，生命得分越小
    return math.exp(k * math.pow(x, 3))

# 车辆信息：车辆编号、车辆所属车站、所属车站坐标、停靠时的时间戳、车辆总运行次数、车辆总运行时间、车辆寿命值
def getInfoOfBike(readfilename, InfoOfBike):
    t = 0
    with open(readfilename, 'r')as rf_obj:
        reader = csv.reader(rf_obj)
        next(reader)
        for row in reader:
            try:
                t += 1
                if row[11] in InfoOfBike:
                    start_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                    finish_time = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
                    driving_time = (finish_time - start_time).seconds / 60
                    if driving_time < 120:
                        InfoOfBike[row[11]][0] = row[7]
                        InfoOfBike[row[11]][1] = float(row[9])
                        InfoOfBike[row[11]][2] = float(row[10])
                        InfoOfBike[row[11]][3] = row[2]
                        InfoOfBike[row[11]][4] += 1
                        InfoOfBike[row[11]][5] += int(driving_time)
                        InfoOfBike[row[11]][6] = WeiBull(InfoOfBike[row[11]][4])
                else:
                    start_time = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
                    finish_time = datetime.strptime(row[2], "%Y-%m-%d %H:%M:%S")
                    driving_time = (finish_time - start_time).seconds / 60
                    if driving_time < 120:
                        InfoOfBike[row[11]] = [row[7], float(row[9]), float(row[10]), row[2], 1, int(driving_time), WeiBull(1)]
            except:
                pass
    print("自行车使用次数: ", t)
    return InfoOfBike

def getInfoOfStation(readfilename, InfoOfStation, InfoOfBike):
    with open(readfilename, 'r')as rf_obj:
        reader = csv.reader(rf_obj)
        next(reader)
        for row in reader:
            if row[3] in InfoOfStation:
                InfoOfStation[row[3]][2] += 1
            else:
                InfoOfStation[row[3]] = [float(row[5]), float(row[6]), 1, 0, [], 0, '2013-07-01 00:00:00']
            if row[7] in InfoOfStation:
                InfoOfStation[row[7]][3] += 1
                InfoOfStation[row[7]][6] = row[2]
            else:
                InfoOfStation[row[7]] = [float(row[9]), float(row[10]), 0, 1, [], 0, row[2]]
    # 为每个车站添加当前车辆编号
    for SID in InfoOfStation:
        InfoOfStation[SID][4] = []
    for bID in InfoOfBike:
        InfoOfStation[InfoOfBike[bID][0]][4].append(bID)
    # 统计可调度的车站容量
    for SID in InfoOfStation:
        InfoOfStation[SID][5] = len(InfoOfStation[SID][4])
    return InfoOfStation


def frequencyDateDicByStation(readfilenamelist):
    FrequencyDateStationMap = {1:{}}
    for readfilename in readfilenamelist:
        with open(readfilename, 'r')as rf_obj:
            reader = csv.reader(rf_obj)
            next(reader)
            for row in reader:
                sta_d = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S").day
                if int(sta_d) not in FrequencyDateStationMap.keys():
                    FrequencyDateStationMap[sta_d] = {}
                start_station_id = row[3]
                if start_station_id in FrequencyDateStationMap[sta_d].keys():
                    FrequencyDateStationMap[sta_d][start_station_id] += 1
                else:
                    FrequencyDateStationMap[sta_d][start_station_id] = 1
                
    for day in FrequencyDateStationMap:
        for key, value in FrequencyDateStationMap[day].items():
            FrequencyDateStationMap[day][key] = int(value/len(readfilenamelist))

    with open("./data/DataBase/FrequencyDateByStation.json", 'w') as json_file:
        json.dump(FrequencyDateStationMap, json_file, ensure_ascii=False)



