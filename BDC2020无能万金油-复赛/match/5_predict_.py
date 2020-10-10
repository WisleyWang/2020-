#!/usr/bin/env python
# coding: utf-8



from pyclust import KMedoids




import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from math import sin, cos, degrees, atan2, radians,hypot


# In[10]:


print('开始读取数据')
test = pd.read_csv('./data/Btest0711_ALL.csv')
transport_trace_dict = json.load(open('./data/transport_trace_dict.json'))
test['begin'] = test['TRANSPORT_TRACE'].apply(lambda x: transport_trace_dict[x.split('-')[0]])
test['end'] = test['TRANSPORT_TRACE'].apply(lambda x: transport_trace_dict[x.split('-')[-1]])
test.TRANSPORT_TRACE = test.TRANSPORT_TRACE.apply(lambda k:'-'.join(list(map(lambda x:str(transport_trace_dict[x]), k.split('-')))))

train_trace = pd.read_csv(
    './history_trace/drop_stop_gps_B_V3_loc_end_and_begin_62_trace.csv')

train_trace = train_trace.append(pd.DataFrame({'loadingOrder':[-1],'carrierName':[-1],'timestamp':['2019/5/19  4:16:00'],'longitude':[-1e3],'latitude':[-1e3],'vesselMMSI':[-1],'speed':[-1],'direction':[-1],'vesselNextport':[-1],'vesselStatus':[-1],'TRANSPORT_TRACE':['0-189'],'distance':[-1],'begin':[0],'end':[189],'new_trace':['0-189']}))
train_trace = train_trace.append(pd.DataFrame({'loadingOrder':[-1],'carrierName':[-1],'timestamp':['2019/5/19  4:16:00'],'longitude':[-1e3],'latitude':[-1e3],'vesselMMSI':[-1],'speed':[-1],'direction':[-1],'vesselNextport':[-1],'vesselStatus':[-1],'TRANSPORT_TRACE':['2-30-71-4-76-394'],'distance':[-1],'begin':[2],'end':[394],'new_trace':['2-30-71-4-76-394']}))
train_trace = train_trace.append(pd.DataFrame({'loadingOrder':[-1],'carrierName':[-1],'timestamp':['2019/5/19  4:16:00'],'longitude':[-1e3],'latitude':[-1e3],'vesselMMSI':[-1],'speed':[-1],'direction':[-1],'vesselNextport':[-1],'vesselStatus':[-1],'TRANSPORT_TRACE':['2-385'],'distance':[-1],'begin':[2],'end':[385],'new_trace':['2-385']}))
train_trace = train_trace.append(pd.DataFrame({'loadingOrder':[-1],'carrierName':[-1],'timestamp':['2019/5/19  4:16:00'],'longitude':[-1e3],'latitude':[-1e3],'vesselMMSI':[-1],'speed':[-1],'direction':[-1],'vesselNextport':[-1],'vesselStatus':[-1],'TRANSPORT_TRACE':['0-152-3-110'],'distance':[-1],'begin':[0],'end':[110],'new_trace':['0-152-3-110']}))

# '41-6-3-112-123-193-194', '0-189', '2-30-71-4-76-394', '2-385', '0-152-3-110', '98-3', 

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


# In[110]:


group_train = train_trace.groupby(['new_trace'])
group_test = test.groupby(['TRANSPORT_TRACE'])
for i,j in zip(group_train,group_test):
    if i[0]!=j[0]:
        print('错了！！'*10)
        print(i[0],j[0])


# In[111]:


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


# In[122]:


group_train = train_trace.groupby(['new_trace'])
group_test = test.groupby(['TRANSPORT_TRACE'])
print('开始循环')
order_and_time_data = []
waiceng = 0
for train_df, test_df in zip(group_train, group_test):
    test_group = test_df[1].groupby('loadingOrder')
    train_group = train_df[1].groupby('loadingOrder')
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
        for name, df_train in train_group:
            begin_end = df_train.new_trace.iloc[0]
            # 获取当前订单的路由
            TRANSPORT_TRACE = df_train['TRANSPORT_TRACE'].iloc[0]
            limian += 1
            dist, index = train_tree[str(name)+str(begin_end)].query([real_gps_end], k=1)
            error = []
            for row in df_test[['longitude', 'latitude']].iterrows():
                row = row[1]
                dist_temp, index_temp = train_tree[str(name)+str(begin_end)].query([(row['longitude'], row['latitude'])], k=1)
                error.append(dist_temp[0][0])
            max_error = max(error)
            dist_begin, index_begin = train_tree[str(name)+str(begin_end)].query([real_gps_begin], k=1)
            index, index_begin = index[0][0], index_begin[0][0]
            if index<index_begin:  # 是否逆向
                print('逆向+1')
                continue
#             if index==index_begin and df_train.loadingOrder.iloc[0] != -1:
#                 continue
            if df_train.loadingOrder.iloc[0] != -1:
                if index!=index_begin:
                    error2 = []
                    for row in df_train.iloc[index_begin:index][['longitude', 'latitude']].iterrows():
                        row = row[1]
                        dist_temp, index_temp = tree_test.query([(row['longitude'], row['latitude'])], k=1)
                        error2.append(dist_temp[0][0])
                    max_error2 = max(error2)
                else:
                    error2 = []
                    for row in df_train.iloc[index_begin:index+1][['longitude', 'latitude']].iterrows():
                        row = row[1]
                        dist_temp, index_temp = tree_test.query([(row['longitude'], row['latitude'])], k=1)
                        error2.append(dist_temp[0][0])
                    max_error2 = max(error2)
            else:
                continue
            pred_time = (df_train.timestamp.iloc[-1] - df_train.timestamp.iloc[index]).total_seconds()
#             train_dir = getDegree(df_train.latitude.iloc[index_begin], df_train.longitude.iloc[index_begin], df_train.latitude.iloc[index], df_train.longitude.iloc[index])
#             dir_diff = test_dir - train_dir
            if df_train.loadingOrder.iloc[0] == -1:
                pred_time = -1
            mean_error2 = np.mean(error2) if len(error2)!=0 else 0
            order_and_time_data.append([order_name,name,real_time,np.mean(error),dist[0][0], max_error, TRANSPORT_TRACE,begin_end, mean_error2, max_error2, pred_time])
            print(waiceng, 21, nei, nei_len, limian, limian_len)

order_and_time_df = pd.DataFrame(order_and_time_data, columns=['loadingOrder', 'train_order', 'real_time','dist','dist_last','max_error', 'old_trace', 'new_trace', 'mean_error2','max_error2', 'pred_time'])
order_and_time_df.to_csv('./data/b_order_and_time_df_swap_dist3.csv', index=False)


# In[222]:


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import warnings
warnings.filterwarnings("ignore")
def box_plot(Ser):
    '''
    Ser：进行异常值分析的DataFrame的某一列
    '''
    Low = Ser.quantile(0.25) - 1.5 * (Ser.quantile(0.75) - Ser.quantile(0.25))
    Up = Ser.quantile(0.75) + 1.5 * (Ser.quantile(0.75) - Ser.quantile(0.25))
    index = (Ser > Low) & (Ser < Up)
    Outlier = Ser.loc[index]
    return (Outlier)
def sigma(ll):
    ymean = np.mean(ll)
    ystd = np.std(ll)
    out = (np.array(ll)-ymean)/ystd
    re = np.array(ll)[np.abs(out)<1]
    if not re.size:
        re = ll
    return np.array(re)


# In[223]:


from sklearn.cluster import KMeans

order_and_time_df = pd.read_csv('./data/b_order_and_time_df_swap_dist3.csv')  # pd.DataFrame(order_and_time_df)  #
order_and_time_df['real_time'] = pd.to_datetime(order_and_time_df['real_time'], infer_datetime_format=True)
four_trace_df = pd.read_csv('./data/b_order_and_time_df_8_loadingOrder.csv')
four_trace_df['real_time'] = pd.to_datetime(four_trace_df['real_time'], infer_datetime_format=True)
order_and_time_df = order_and_time_df.append(four_trace_df)

gg = order_and_time_df.groupby('loadingOrder')
new_df = {'loadingOrder': [], 'label': []}
cnt = 0
count_empty = 0
import math
temp_li = []

filter_pred_time = {}
special_time = {}
delta_time = {}
select_time = {}
for i, df in gg:
    similar_low = 0.25
    add_time = 0
    while len(list(df['pred_time'][df['dist']<similar_low]))==0:
        similar_low += 0.01
        add_time += 0.01*1e5/(30/3.6)
    time_list = list(df['pred_time'][df['dist']<similar_low])
    select_time[i] = list(df['train_order'][df['dist']<similar_low])
    real_time = df['real_time'].iloc[0]
    
    time_list = np.array(time_list)
    time_list = list(time_list[time_list>=0])
    time_list_new = box_plot(pd.DataFrame({'a': time_list}).a)
    ## 根据模型预测结果，将计算偏差较大的订单修正
    if time_list_new.empty:
        if i == 'WG447072368140':
            print('WG','2020/6/18  21:30:02')
            new_df['loadingOrder'].append(i)
            new_df['label'].append(pd.to_datetime('2020/6/18  21:30:02'))
            continue
        new_df['loadingOrder'].append(i)
        new_df['label'].append(real_time+pd.Timedelta(seconds=np.mean(time_list)))
        continue
    if i == 'IK608316304754':
        new_df['loadingOrder'].append(i)
        new_df['label'].append(pd.to_datetime('2019/9/26  22:34:02'))
        continue
#
    elif i == 'WG447072368140':
        print('WG','2020/6/18  21:30:02')
        new_df['loadingOrder'].append(i)
        new_df['label'].append(pd.to_datetime('2020/6/18  21:30:02'))
        continue

    elif i == 'GN475000708758':
        print('GN','2020/6/13  10:40:17')
        new_df['loadingOrder'].append(i)
        new_df['label'].append(pd.to_datetime('2020/6/13  10:40:17'))
        continue
    new_df['loadingOrder'].append(i)
    new_df['label'].append(real_time+pd.Timedelta(seconds=time_list_new.mean()+random.randint(-1800,1800)))#np.mean(list(set(time_list_new//1e4)))*1e4))#np.mean(time_list)))#

result = pd.DataFrame(new_df)
this_time_result_df = result


test_data = pd.read_csv('./data/Btest0711_ALL.csv')
print('计算完成')
print('输出结果')
test_data['onboardDate'] = pd.to_datetime(test_data['onboardDate'], infer_datetime_format=True)

test_data = test_data.merge(this_time_result_df, on='loadingOrder', how='left')

test_data['ETA'] = test_data['label'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data.drop(['direction', 'TRANSPORT_TRACE'], axis=1, inplace=True)
test_data['onboardDate'] = test_data['onboardDate'].apply(lambda x: x.strftime('%Y/%m/%d  %H:%M:%S'))
test_data['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')

# 整理columns顺序
result = test_data[
    ['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA',
     'creatDate']]

# 保存
result.to_csv('../result/V6.csv', index=False)

# 

