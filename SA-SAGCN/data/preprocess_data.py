
import pandas as pd
import numpy as np
import json
import time
import os

data_file_list = [
    r'.\rowData\2014-04 - Citi Bike trip data.csv',
    r'.\rowData\2014-05 - Citi Bike trip data.csv',
    r'.\rowData\2014-06 - Citi Bike trip data.csv',
    r'.\rowData\2014-07 - Citi Bike trip data.csv',
    ]


def get_station_latlon():
    if not os.path.exists("./data/DataBase/station_latlon.json"):
        all_station = {}
        single_dataframe = pd.read_csv(data_file_list[0],usecols=[3,5,6,7,9,10])
        # 对dataframe中的数据按行进行遍历
        for index,row in single_dataframe.iterrows():
            if int(row['start station id']) not in all_station:
                all_station[int(row['start station id'])] = [row['start station latitude'],row['start station longitude']]
            if int(row['end station id']) not in all_station:
                all_station[int(row['end station id'])] = [row['end station latitude'], row['end station longitude']]
        # 车站及坐标数据保存到json文件中
        json_str = json.dumps(all_station)
        with open('./data/DataBase/station_latlon.json','w') as json_file:
            json_file.write(json_str)
        # print(len(all_station))# 330
        return all_station

    with open("./data/DataBase/station_latlon.json", "r") as json_file:
        all_station = json.load(json_file)
    all_station = list(map(int, all_station.keys()))
    return all_station

def clean_data():
    all_station = get_station_latlon()
    all_data_frames = []
    for data_file in data_file_list:
        df = pd.read_csv(data_file, usecols=[1,2,3,5,6,7,9,10,11])
        df.dropna(axis=0,how='any',inplace=True)
        df['starttime'] = pd.to_datetime(df['starttime'])
        df['stoptime'] = pd.to_datetime(df['stoptime'])
        df['differ_time'] = round((df['stoptime'] - df['starttime']).dt.seconds/60, 2)
        df = df.drop(df[(df["starttime"].dt.hour>=0) & ( df["starttime"].dt.hour<7)].index)
        df = df.drop(df[(df["stoptime"].dt.hour>=0) & (df["stoptime"].dt.hour<7)].index)
        df = df.drop(df[(df['differ_time']>120)|(df['differ_time']<1)].index)
        df = df.drop(df[(~df['start station id'].isin(all_station)) | (~df['end station id'].isin(all_station))].index)
        # 将清洗后的新数据保存在新的文件中
        all_data_frames.append(df)
    data_frame_concat = pd.concat(all_data_frames,axis=0)
    # print(len(data_frame_concat))  # 3410899
    return data_frame_concat


def get_time_idx(time):
    # 2014-04-01 00:00:07
    time = time.strftime
    mon = int(time[5:7])
    day = int(time[8:10])
    hour = int(time[11:13])
    minu = int(time[14:16])//20
    
    count =  day * 3 * 24 + hour * 3 + minu - 72
    if mon==5:
        count += 30 * 3 * 24
    elif mon==6:
        count += 61 * 3 * 24
    elif mon==7:
        count += 91 * 3 * 24
    return count


df = clean_data()
df["starttime idx"] = df["starttime"].apply(get_time_idx)
df_agg = df.groupby(['starttime idx', "start station id"]).agg(["count"])
df_agg = df_agg.iloc[:,0].sort_index()


time = set()
station = set()
for t, s in df_agg.index:
    time.add(t)
    station.add(s)
N = len(station)
len(time), len(station)


ar = np.zeros(shape=(len(time), len(station))).flatten()
for v in df_agg.values:
    ar[v%N] = v
ar = ar.reshape(-1, N, 1) * 7
print(ar.shape)
# ar.shape # (8784, 330, 1)
np.save("./data/DataBase/BT_Data.npy", ar)

