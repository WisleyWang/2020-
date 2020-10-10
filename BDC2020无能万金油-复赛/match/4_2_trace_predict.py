#!/usr/bin/env python
# coding: utf-8

# In[46]:



# In[1]:


from pyclust import KMedoids


# In[1]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from math import sin, cos, degrees, atan2, radians,hypot


# In[2]:


print('开始读取数据')
test = pd.read_csv('./data/Btest0711_ALL.csv')
transport_trace_dict = json.load(open('./json_data/transport_trace_dict.json'))
test['begin'] = test['TRANSPORT_TRACE'].apply(lambda x: transport_trace_dict[x.split('-')[0]])
test['end'] = test['TRANSPORT_TRACE'].apply(lambda x: transport_trace_dict[x.split('-')[-1]])
test.TRANSPORT_TRACE = test.TRANSPORT_TRACE.apply(lambda k:'-'.join(list(map(lambda x:str(transport_trace_dict[x]), k.split('-')))))

train_trace = pd.read_csv(
    './history_trace/B_4_no_history_trace.csv')
test = test[test.TRANSPORT_TRACE.isin(train_trace.new_trace)]

print('开始处理时间')
train_trace['timestamp'] = pd.to_datetime(train_trace['timestamp'], infer_datetime_format=True)
test['timestamp'] = pd.to_datetime(test['timestamp'], infer_datetime_format=True)
test['onboardDate'] = pd.to_datetime(test['onboardDate'], infer_datetime_format=True)
test['onboardDate'] = test['onboardDate'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
test['onboardDate'] = pd.to_datetime(test['onboardDate'], infer_datetime_format=True)
print('处理时间完成')
print('开始排序分组')

train_trace.sort_values(['new_trace', 'timestamp'], inplace=True)
test.sort_values(['TRANSPORT_TRACE', 'timestamp'], inplace=True)


# In[3]:


group_train = train_trace.groupby(['new_trace'])
group_test = test.groupby(['TRANSPORT_TRACE'])
for i,j in zip(group_train,group_test):
    if i[0]!=j[0]:
        print(i[0],j[0])


# In[71]:


port_lonlat = {'20': (50.191041999999996, 26.479906), '71': (79.891802, 6.984864999999999), '148': (4.885655, 43.430234000000006), '10': (139.792252, 35.616509),
 '79': (23.616167, 37.971821999999996), '3': (103.70461999999999, 1.3031409999999999), '24': (1.313896, 51.963177), '288': (-88.19550600000001, 17.507049),
 '11': (32.301866, 31.265289000000003), '38': (-0.327021, 39.460366), '21': (56.737261, 24.367309), '4': (55.04979, 25.022073000000002),
 '0': (114.275347, 22.5777), '26': (-46.28402, -23.954513000000002), '2': (113.86305800000001, 22.559462), '77': (67.333229, 24.774487),
 '16': (13.728527, 45.537061), '147': (174.78561399999998, -36.844873), '41': (113.653433, 22.694213), '6': (103.545456, 1.399416),
 '67': (114.13970900000001, 22.419915), '312': (130.41038600000002, 33.660845), '46': (-5.8129800000000005, 35.788207), '110': (0.204605, 49.490140000000004),
 '28': (121.046856, 13.756824), '19': (-123.103178, 49.312003999999995), '22': (67.32748000000001, 24.766207), '394': (48.154646, 29.045214),
 '44': (106.881179, -6.119484), '159': (3.064607, 36.768454999999996), '105': (-79.88299, 9.352609), '9': (114.167519, 22.246522),
 '39': (14.509457000000001, 35.896571), '194': (9.906216, 2.939002), '96': (91.79332099999999, 22.304439000000002), '111': (-1.121292, 50.790419),
 '60': (29.748835999999997, 40.7739), '103': (8.946256, 44.40565), '30': (72.93839, 18.936253), '189': (14.427023000000002, 45.342132),
 '31': (31.050079999999998, -29.868304), '47': (-7.607741000000001, 33.599299), '153': (131.882303, 43.114035), '139': (126.626175, 37.401819),
 '7': (4.6026739999999995, 51.90936), '25': (-71.642993, -33.030843), '29': (101.361204, 3.034709), '88': (113.8702, 22.538085),
 '385': (-71.609932, -33.578213), '232': (50.618590000000005, 26.135392), '101': (100.88608, 13.079153), '118': (-69.638318, 18.425821),
 '87': (103.400193, 10.888632000000001), '1': (-104.305571, 19.085960999999998), '98': (121.64364599999999, 31.344022), '409': (-23.494260999999998, 14.919073000000001), '43': (-58.35677, -34.627862)}
def getDegree(latA, lonA, latB, lonB):
    """
    由两个坐标点的经纬度计算航向
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return int(brng*100)
import time


# In[77]:


group_train = train_trace.groupby(['new_trace'])
group_test = test.groupby(['TRANSPORT_TRACE'])
print('开始循环')
order_and_time_data = []
waiceng = 0
for train_df, test_df in zip(group_train, group_test):
    test_group = test_df[1].groupby('loadingOrder')
    nice_train_df = train_df[1][train_df[1].TRANSPORT_TRACE.isin([test_df[0]])].copy()
    porr_train_df = train_df[1][train_df[1].TRANSPORT_TRACE.isin([test_df[0]])==False].copy()
    train_group = porr_train_df.groupby('loadingOrder')
    nice_train_df_gf = nice_train_df.groupby('loadingOrder')
    waiceng += 1
    train_tree = {}

    print('开始创建树')
    for order_name, df_train in train_group:
        gf = df_train.groupby(['new_trace'])
        for begin_end, df_train in gf:
            lon_lat = np.array([(i, j) for i, j in zip(df_train['longitude'], df_train['latitude'])])
            tree = KDTree(lon_lat, leaf_size=2)
            train_tree[str(order_name)+str(begin_end)] = tree

    print('创建树完成')
    nei = 0
    nei_len = len(test_group)
    for order_name, df_test in test_group:
        lon_lat_test = np.array([(i, j) for i, j in zip(df_test['longitude'], df_test['latitude'])])
        tree_test = KDTree(lon_lat_test, leaf_size=2)
        ##########这里错了还没改！！！！！！！！！！！！！！！！！已经改了！！！！！！！！！！
        real_gps_end = (df_test.longitude.iloc[-1], df_test.latitude.iloc[-1])
        real_gps_begin = (df_test.longitude.iloc[0], df_test.latitude.iloc[0])
#         test_dir = getDegree(real_gps_begin[1], real_gps_begin[0], real_gps_end[1], real_gps_end[0])
        real_time =  df_test.timestamp.iloc[-1]
        nei += 1
        limian = 0
        limian_len = len(train_group)
        test_df_list = []
        test_df_list.append(df_test)
#         keep_nic
        cnt_len = 0
        nice_df_temp = None
        for nice, nice_df in nice_train_df_gf:
            if len(nice_df)>cnt_len:
                nice_df_temp = nice_df
                cnt_len = len(nice_df)
        nice_df = nice_df_temp
        lon_lat_test = np.array([(i, j) for i, j in zip(nice_df['longitude'], nice_df['latitude'])])
        tree_nice_df = KDTree(lon_lat_test, leaf_size=2)
        dist_begin, index_begin = tree_nice_df.query([real_gps_end], k=1)
        if dist_begin[0][0]<2:
            nice_begin = real_gps_end
            nice_df = nice_df[index_begin[0][0]:]
        else:
            nice_begin = (nice_df.longitude.iloc[0], nice_df.latitude.iloc[0])

#             nice_begin = (nice_df.longitude.iloc[0], nice_df.latitude.iloc[0])
        len_dd = len(nice_df)
        index = len_dd
        drop_dist = 0
        # %% 根据末尾速度判断是否要停止前行
        for speed in nice_df['speed'].values[::-1]:
            if speed < 5 and drop_dist<25000:
                index -= 1
                drop_dist+=nice_df['distance'].iloc[index]
            else:
                break
        if index < 2:
            continue

        t = (nice_df.timestamp.iloc[index-1] - nice_df.timestamp.iloc[0]).total_seconds()
        test_df_list.append(nice_df.iloc[:index].copy())
        for name, df_train in train_group:
            begin_end = df_train.new_trace.iloc[0]
            # 获取当前订单的路由
            TRANSPORT_TRACE = df_train['TRANSPORT_TRACE'].iloc[0]
            limian += 1
            dist, index = train_tree[str(name)+str(begin_end)].query([real_gps_end], k=1)

            dist_begin, index_begin = train_tree[str(name)+str(begin_end)].query([nice_begin], k=1)
            index, index_begin = index[0][0], index_begin[0][0]
            if test_df[0] == '2-385':
                if dist_begin[0][0] <6 and dist[0][0]<6:
                    pred_time = (df_train.timestamp.iloc[index_begin] - df_train.timestamp.iloc[index]).total_seconds()+t
                    test_df_list.append(df_train.iloc[index:index_begin].copy())
                else:
                    continue
            else:
                if dist_begin[0][0] <2 and dist[0][0]<2:
                    pred_time = (df_train.timestamp.iloc[index_begin] - df_train.timestamp.iloc[index]).total_seconds()+t
                    test_df_list.append(df_train.iloc[index:index_begin].copy())
                else:
                    continue
            print(test_df[0])
            max_error = 0
            mean_error = 0
            max_error2 = 0
            mean_error2 = 0
            if df_train.loadingOrder.iloc[0] == -1:
                pred_time = -1
            order_and_time_data.append([order_name,name,real_time,mean_error,dist[0][0], max_error, TRANSPORT_TRACE,begin_end,mean_error2, max_error2 , pred_time])
            print(waiceng, 21, nei, nei_len, limian, limian_len)
#             break #time.time()*1e3
        pd.concat(test_df_list).to_csv('./special_file/b_nohistory_trace/'+order_name+'.csv')
order_and_time_df = pd.DataFrame(order_and_time_data, columns=['loadingOrder', 'train_order', 'real_time','dist','dist_last','max_error', 'old_trace', 'new_trace', 'mean_error2','max_error2', 'pred_time'])
order_and_time_df.to_csv('./data/b_order_and_time_df_8_loadingOrder.csv', index=False)

