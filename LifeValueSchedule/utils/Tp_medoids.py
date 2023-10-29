import random
import numpy as np
import copy
import folium
import matplotlib.pyplot as plt

distances_cache = {}
MAX_CLUSTER = 12
kesi = 0.04  # 距离阈值

def importData(FrequencyStationLaLo):
    FSLaLoData = list(FrequencyStationLaLo.values())
    LaLoData = []
    for i in FSLaLoData:
        LaLoData.append([i[0], i[1], i[2] - i[3]])
    StationNo = list(FrequencyStationLaLo.keys())
    return StationNo, LaLoData

def Manhattan_Distance(station1, station2):
    M_Distance = abs(station1[0] - station2[0]) + abs(station1[1] - station2[1])
    return M_Distance


def get_throughput_recluster(InfoOfStation, medoids_idx):
    """
    StationLaLo：经纬度和吞吐量
    medoids_idx：聚类中心结点
    """
    medoids_station = {}
    medoids_throughput = [0] * len(medoids_idx)
    not_choosed_station = []

    for idx, center in enumerate(medoids_idx):
        medoids_station[center] = []
        medoids_throughput[idx] += InfoOfStation[center][2] - InfoOfStation[center][3]

    for station, staInfo in InfoOfStation.items():
        if station in medoids_idx:
            continue
        
        min_dist = np.inf
        choice = None
        for idx, center in enumerate(medoids_idx):
            tmp_dis = Manhattan_Distance(staInfo, InfoOfStation[center])
            if tmp_dis < kesi and tmp_dis < min_dist:
            # if tmp_dis < min_dist:
                min_dist = tmp_dis
                choice = idx
        if choice is not None:
            medoids_station[medoids_idx[choice]].append(station)
            medoids_throughput[choice] += InfoOfStation[station][2] - InfoOfStation[station][3]
        else:
            # print("Not Choice a Medoid!")
            not_choosed_station.append(station)

    if(len(not_choosed_station)):
        print("Break Kesi Limit Station Num: ", len(not_choosed_station))
        for station in not_choosed_station:
            min_dist = np.inf
            choice = None
            for idx, center in enumerate(medoids_idx):
                tmp_dis = Manhattan_Distance(InfoOfStation[station], InfoOfStation[center])
                if tmp_dis<min_dist:
                    choice = idx
                    min_dist = tmp_dis
            medoids_station[medoids_idx[choice]].append(station)
            medoids_throughput[choice] += InfoOfStation[station][2] - InfoOfStation[station][3]

    Throughput = 0
    for m in range(len(medoids_idx)):
        Throughput += abs(medoids_throughput[m])
    return Throughput, medoids_station, not_choosed_station


def Tp_medoids(InfoOfStation, k):
    medoids_idx = random.sample(InfoOfStation.keys(), k)
    pre_throughput, medoids_cluster, not_choosed_station = get_throughput_recluster(InfoOfStation, medoids_idx)

    best_medoids_cluster = copy.copy(medoids_cluster)
    iteration = 1
    patience = 0
    # 更新中间结点
    while True:
        tmp_medoids_cluster = {}
        for center, medoid_station in medoids_cluster.items():
                all_station = [center] + medoid_station                
                new_center = center
                min_dis = np.inf
                for i in all_station:
                    disi = 0
                    for j in all_station:
                        if i!=j:
                            disi += Manhattan_Distance(InfoOfStation[i], InfoOfStation[j])
                        
                    if disi < min_dis:
                        min_dis = disi
                        new_center = i

                all_station.remove(new_center)
                tmp_medoids_cluster[new_center] = all_station

        tmp_throughput, tmp_medoids_cluster, not_choosed_station = get_throughput_recluster(InfoOfStation, list(tmp_medoids_cluster.keys()))

        if patience >= 5 and set(medoids_cluster.keys()) == set(tmp_medoids_cluster.keys()):
            break

        medoids_cluster = copy.copy(tmp_medoids_cluster)

        # 根据 throughput 是否要更新集群中心结点
        if tmp_throughput < pre_throughput:
            pre_throughput = tmp_throughput
            best_medoids_cluster = copy.copy(tmp_medoids_cluster)
            patience = 0

        iteration += 1
        patience += 1

    print("Tp_medoids Iteration Count: ", iteration)
    return best_medoids_cluster, pre_throughput, iteration, not_choosed_station


def dis2center(InfoOfStation, best_medoids_cluster):
    distance = 0
    for center, medoids in best_medoids_cluster.items():
        for m in medoids:
           distance += Manhattan_Distance(InfoOfStation[m], InfoOfStation[center])
    return distance

def DrawingofClusterResult(Kmresult, InfoOfStation, not_choosed_station):
    COLOUR = ['red', 'blue', 'lime', 'yellow', 'orange', 'green', 'pink', 'grey', 'brown', 'cyan', 'purple', 'olive']
    s3 = None
    for t, center in enumerate(Kmresult):
        cc = COLOUR[t]
        s1 = plt.scatter(InfoOfStation[center][1], InfoOfStation[center][0], c=cc, marker="*")
        for s in Kmresult[center]:
            s2 = plt.scatter(InfoOfStation[s][1], InfoOfStation[s][0], c=cc)
    
    for station in not_choosed_station:
        s3 = plt.scatter(InfoOfStation[station][1], InfoOfStation[station][0], c='black', marker=".")

    if s3:
        plt.legend((s1,s2, s3), ('Center Station','Cluster Station', 'Abnormal Station'), loc='upper left')
    else:
        plt.legend((s1,s2), ('Center Station','Cluster Station'), loc='upper left')
    plt.ylim(top=40.79)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Station Clustering Results with Kesi {:.3f}".format(kesi))
    plt.show()
    plt.close()


def stationclustermap(Kmresult, InfoOfStation):
    COLOUR = ['red', 'blue', 'black', 'yellow', 'orange', 'green', 'maroon', 'navy', 'brown', 'cyan', 'purple', 'teal']
    map_osm = folium.Map(location=[40.7143528, -74.0059731], zoom_start=14, tiles='Stamen Terrain')

    for t, center in enumerate(Kmresult):
        for m in Kmresult[center]:
            folium.RegularPolygonMarker(InfoOfStation[m][:2], fill_color=COLOUR[t], number_of_sides=10, radius=7).add_to(
                map_osm)

    map_osm.save(r'./StationMap.html')


def Tp_medoids_main(InfoOfStation):
    best_medoids_cluster, throughput, iteration, not_choosed_station = Tp_medoids(InfoOfStation, MAX_CLUSTER)
    print("Tp_medoids Throughput: ", throughput)

    # all_dis = dis2center(InfoOfStation, best_medoids_cluster)
    # print("Tp_medoids Distance of All Nodes to Center:", all_dis)

    # # 热力图显示聚类结果
    # stationclustermap(best_medoids_cluster, InfoOfStation)
    # DrawingofClusterResult(best_medoids_cluster, InfoOfStation, not_choosed_station)
    return best_medoids_cluster

# # Kesi 是聚类距离变量
# epochs = 10
# kesi_arr = np.linspace(0.01, 0.06, 10)
# for epoch in range(epochs):
#     kesi = kesi_arr[epoch]
#     K_mediods_cluster = Tp_medoids(InfoOfStation)



