#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import json
import warnings
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
warnings.filterwarnings("ignore")

'''
本代码和代码4差不多，本代码将订单的尾部已到港部分砍掉（判断在港口逗留）
'''


# In[23]:


# train_dir = './history_trace/test_data_port_super_normal_trace_V2_all_col_add_bothway.csv'
train_dir = './data/test_data_normal_trace.csv'
#                             test_data_port_super_perfect_normal_trace_B_V1_66_trace
train_data = pd.read_csv(train_dir)
# test_data = pd.read_csv('./data/R2 ATest 0711.csv')
test_data = pd.read_csv('./data/Btest0711_ALL.csv')
print('读取文件完成')
train_data.speed[train_data.speed>60] = 60
train_data.direction[train_data.direction>35999] = 0


train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], infer_datetime_format=True)

train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
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


# In[24]:


transport_trace_dict = json.load(open('./data/transport_trace_dict.json', 'r'))

test_list = ['CNSHK-SGSIN', 'CNSHK-MYTPP', 'CNSHK-BHBAH', 'CNSHK-EGPSD',       'CNYTN-BRSSZ', 'CNNSA-MYTPP-SGSIN-ZACPT-CGPNR-GALBV-CMKBI',
       'CNSHK-SGSIN-MYPKG', 'CNYTN-SGSIN-EGSUZ-GRPIR-ITSPE-ITGOA',       'CNYTN-BDCGP', 'CNSHK-MYTPP-MUPLU-ZADUR', 'CNYTN-MXZLO',
       'CNSHK-THLCH', 'CNSHK-IDJKT', 'CNYTN-NZAKL', 'CNDCB-SGSIN',       'CNSHK-KRINC', 'CNNSA-SGSIN-AEJEA', 'CNSHK-MYPKG', 'CNYTN-HRRIJ',
       'CNSHK-SGSIN-MYPKG-INMUN-PKKHI-PKBQM', 'CNSHK-JPTYO',       'CNSHK-HKHKG-TWKHH-CNNBG-CNSHA-CNTAO-KRPUS-MXZLO-PABLB-PAMIT-COCTG-JMKIN-DOCAU',
       'CNYTN-CAVAN', 'CNSHK-SGSIN-MTMLA-DZALG', 'CNYTN-PHBTG',       'CNSHK-INNSA-LKCMB-AEJEA-AEAUH-KWKWI', 'CNSHK-INNSA',
       'SGSIN-GRPIR', 'CNSHK-CLVAP', 'CNYTN-ARENA', 'SIKOP-HKHKG',       'CNSHK-GRPIR', 'BRSSZ-CNSHK', 'CNSHK-SGSIN-AEJEA-QAHMD-SADMM',
       'CNYTN-GBFXT', 'CNYTN-PAONX', 'CNSHK-KHKOS',       'CNSHK-SGSIN-MTMLA-SIKOP', 'CNHKG-RUVVO',
       'CNYTN-MYTPP-EGSUE-MATNG-MACAS', 'CNSHK-CLSAI', 'CNSHK-JPHKT',       'CNYTN-ESVAL', 'CNYTN-VNVUT-SGSIN-FRLEH', 'CNYTN-MATNG',
       'CNYTN-SGSIN-MTMLA-DZALG', 'CNYTN-GRPIR', 'CNSHK-TRYAR',       'CNSHK-ESVAL', 'CNYTN-MTMLA', 'CNSHK-LKCMB', 'CNSHK-BZBZE',
       'CNSHK-PKQCT', 'CNYTN-SGSIN', 'CNSHK-SIKOP',       'CNSHK-GRPIR-ITSPE-ITGOA', 'CNSHK-ZADUR', 'CNYTN-CVRAI',
       'CNYTN-MYTPP', 'CNSHA-SGSIN',       'CNSHK-SGSIN-AEJEA-QAHMD-SADMM-OMSOH', 'CNNSA-CMKBI',       'HKHKG-FRFOS', 'CNYTN-GBSOU', 'CNYTN-NLRTM', 'CNSHK-SGSIN-AEJEA']
# test_list = ['BRSSZ-CNSHK', 'CNDCB-SGSIN', 'CNHKG-RUVVO', 'CNNSA-AEJEA',
#  'CNNSA-CMKBI', 'CNSHA-SGSIN', 'CNSHK-AEJEA', 'CNSHK-BHBAH', 'CNSHK-BZBZE', 'CNSHK-CLSAI', 'CNSHK-CLVAP', 'CNSHK-DOCAU',
#  'CNSHK-DZALG', 'CNSHK-EGPSD', 'CNSHK-ESVAL', 'CNSHK-GRPIR', 'CNSHK-IDJKT', 'CNSHK-INNSA', 'CNSHK-ITGOA', 'CNSHK-JPHKT',
#  'CNSHK-JPTYO', 'CNSHK-KHKOS', 'CNSHK-KRINC', 'CNSHK-KWKWI', 'CNSHK-LKCMB', 'CNSHK-MYPKG', 'CNSHK-MYTPP', 'CNSHK-OMSOH',
#  'CNSHK-PKBQM', 'CNSHK-PKQCT', 'CNSHK-SADMM', 'CNSHK-SGSIN', 'CNSHK-SIKOP', 'CNSHK-THLCH', 'CNSHK-TRYAR', 'CNSHK-ZADUR',
#  'CNYTN-ARENA', 'CNYTN-BDCGP', 'CNYTN-BRSSZ', 'CNYTN-CAVAN', 'CNYTN-CVRAI', 'CNYTN-DZALG', 'CNYTN-ESVAL', 'CNYTN-FRLEH',
#  'CNYTN-GBFXT', 'CNYTN-GBSOU', 'CNYTN-GRPIR', 'CNYTN-HRRIJ', 'CNYTN-ITGOA', 'CNYTN-MACAS', 'CNYTN-MATNG', 'CNYTN-MTMLA',
#  'CNYTN-MXZLO', 'CNYTN-MYTPP', 'CNYTN-NLRTM', 'CNYTN-NZAKL', 'CNYTN-PAONX', 'CNYTN-PHBTG', 'CNYTN-SGSIN', 'HKHKG-FRFOS',
#  'SGSIN-GRPIR', 'SIKOP-HKHKG']
# test_list = [[str(transport_trace_dict[i.split('-')[0]]), str(transport_trace_dict[i.split('-')[-1]])] for i in
#              test_list]
# test_data.TRANSPORT_TRACE = test_data.TRANSPORT_TRACE.apply(lambda x:)
# test_list = [list(map(lambda x:str(transport_trace_dict[x]), trace.split('-'))) for trace in test_list]


new_test_list = []
for i in test_list:
    if i not in new_test_list:
        new_test_list.append(i)
test_list = new_test_list


# In[25]:


train_data.new_trace.unique().__len__()


# In[26]:


test_list.__len__()


# In[28]:


train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
group_df = train_data.groupby('loadingOrder')
df_list = []
df_cat_list = []
len_df = len(group_df)
file_cnt = 0
cnt = 0
# need = ['305-284', '0-6-46-7-104', '98-74', '0-67-147-155', '41-6-3-112-123-193-194', '0-189', '0-152-3-110', '2-30-71-4-76-394']
for order_name, df in group_df:
    gf = df.groupby('new_trace')
    for begin_end, df in gf:
        len_dd = len(df)
        index = len_dd
        drop_dist = 0
        # %% 根据末尾速度判断是否要停止前行
        for speed in df['speed'].values[::-1]:
            if speed < 5 and drop_dist<70000:
                index -= 1
                drop_dist+=df['distance'].iloc[index]
            else:
                break
        if index < 2:
            continue
        df = df.iloc[:index]

        # %% 重新对路径进行截取
        lon_lat = np.array([(i, j) for i, j in zip(df['longitude'], df['latitude'])])
        tree = KDTree(lon_lat, leaf_size=2)

        trace = df['new_trace'].iloc[0].split('-')
        dist_begin, index_begin = tree.query([port_lonlat[trace[0]]], k=1)
        dist_end, index_end = tree.query([port_lonlat[trace[-1]]], k=1)

        # 排除掉定位差距过大的路径
        if dist_end[0][0] * 1e5 > 70 * 1e3 or dist_begin[0][0] * 1e5 > 70 * 1e3:
            cnt += 1
            if trace[0] == '98'and trace[-1] == '3':  # 保留特殊路径
                pass
#             elif begin == '305'and end == '284':
#                 pass
# #             #elif:2 272
# #             #elif:41 194
            else:
                continue

#         index_begin = index_begin[0][0]
        index_end = index_end[0][0] 
        df_list.append(df.iloc[:index_end + 1])#index_begin

        file_cnt += 1
        print("第%d个文件" % 0, '第%d组' % file_cnt, len_df)


dd = pd.concat(df_list)
temp_dd = train_data[train_data.new_trace.isin(['41-194'])]#41-6-3-112-123-193-194

temp_dd.new_trace = '41-6-3-112-123-193-194'

dd = dd.append(temp_dd)
dd.new_trace.unique().__len__()

dd.to_csv(
    './history_trace/drop_stop_gps_B_V3_loc_end_and_begin_62_trace.csv',
    index=False)

