import copy
import numpy as np
import json

def StationDemand():
    # 获得车站的实际需求
    with open("./data/DataBase/frequencyDateByStation.json", 'r') as fp:
        station_demand = json.load(fp)
    for key, value in station_demand.items():
        station_demand[key] = sorted(value.items(), key=lambda x: x[1])
    return station_demand

def BikeNormalized(bif):
    bkAppl_dict = {}
    for bkinfo in bif:
        bkAppl_dict[bkinfo] = bif[bkinfo][6]
    bkAppl_list = sorted(bkAppl_dict.items(), key=lambda item: item[1])
    return bkAppl_list

def MaxMinschWithDemand(sif, bif, cif, stRequ, bkLife, day_number):
    # 车辆信息：车辆编号、车辆所属车站、所属车站坐标、停靠时的时间戳、车辆总运行次数、车辆总运行时间、车辆寿命值
    # 车站信息：车站号、车站坐标、车站进量、车站出量、车站停靠车辆、当前可调度车辆、当前时间戳

    sif_new = copy.deepcopy(sif)
    bif_new = copy.deepcopy(bif)

    station_demand = {}
    for station in sif.keys():
        station_demand[station] = 0
    for station, req in stRequ[str(day_number)]:
        station_demand[station] = req
    station_demand = sorted(station_demand.items(),key = lambda x:x[1])

    for clu in cif:
        clu_station = [clu] + cif[clu]
        clu_station_demand = {}
        clu_bike_id = []
        clu_bike_life = {}

        # 在全部车站需求中选出当前集群的车站和其中的车辆
        for station, demand in station_demand:
            if station in clu_station:
                clu_bike_id += sif[station][4]
                clu_station_demand[station] = demand
        for bike, life in bkLife:
            if bike in clu_bike_id:
                clu_bike_life[bike] = life
        
        j = 0  # 作为车子总数的统计
        bike_num = len(clu_bike_life)  # 总共该集群拥有的车子的数量

        demand_num = len(clu_station_demand.values())
        for station, demand in clu_station_demand.items():
            demand = int(bike_num * demand/demand_num)  # 针对自行车数量和站点需求做归一化
            if j < bike_num:  # 集群内车子数量充足
                temp = list(clu_bike_life.keys())[j:j+demand]
                for bike_id in temp: # 选取 demand 个车子    
                    bif_new[bike_id][0] = station
                    bif_new[bike_id][1] = sif_new[station][0]
                    bif_new[bike_id][2] = sif_new[station][1]
                sif_new[station][4] = temp
                sif_new[station][5] = len(temp)
                j = min(j+demand, bike_num)
            elif j>=bike_num:  # 车子数量不够
                sif_new[station][4] = []
                sif_new[station][5] = 0

        if j < bike_num:  # 剩余自行车，放在需求量最大的车站中
            temp = list(clu_bike_life.keys())[j:bike_num]
            for bike_id in temp:
                bif_new[bike_id][0] = station
                bif_new[bike_id][1] = sif_new[station][0]
                bif_new[bike_id][2] = sif_new[station][1]
            sif_new[station][4] = temp
            sif_new[station][5] = len(temp)       

    return sif_new, bif_new

def Scheduling(bif, sif, cif, day_number):
    stRequ = StationDemand()
    bkLife = BikeNormalized(bif) 
    sif_new, bif_new = MaxMinschWithDemand(sif, bif, cif, stRequ, bkLife, day_number)
    return sif_new, bif_new

