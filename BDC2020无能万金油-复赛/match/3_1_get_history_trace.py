#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
import json
import warnings
warnings.filterwarnings("ignore")

"""
本代码用于提取历史路径，根据测试集的港口来定位，截取历史路径
"""


# In[4]:



transport_trace_dict = json.load(open('./data/transport_trace_dict.json', 'r'))

# %% 提取port。csv中需要用的港口坐标
port_df = pd.read_csv('./data/port.csv')
port_df.iloc[0] = port_df.iloc[0].apply(
    lambda x: str(x).replace(' ', '').replace('<', '').replace('>', '').replace('$', ''))

# 测试集出现过的港口
test_port = ['CNSHK', 'SGSIN', 'CNSHK', 'MYTPP', 'CNSHK', 'BHBAH', 'CNSHK', 'EGPSD', 'CNYTN', 'BRSSZ', 'CNNSA', 'CMKBI', 'CNSHK', 'MYPKG', 'CNYTN', 'ITGOA', 'CNYTN', 'BDCGP', 'CNSHK', 'ZADUR', 'CNYTN', 'MXZLO', 'CNSHK', 'THLCH', 'CNSHK', 'IDJKT', 'CNYTN', 'NZAKL', 'CNDCB', 'SGSIN', 'CNSHK', 'KRINC', 'CNNSA', 'AEJEA', 'CNSHK', 'MYPKG', 'CNYTN', 'HRRIJ', 'CNSHK', 'PKBQM', 'CNSHK', 'JPTYO', 'CNSHK', 'DOCAU', 'CNYTN', 'CAVAN', 'CNSHK', 'DZALG', 'CNYTN', 'PHBTG', 'CNSHK', 'KWKWI', 'CNSHK', 'INNSA', 'SGSIN', 'GRPIR', 'CNSHK', 'CLVAP', 'CNYTN', 'ARENA', 'SIKOP', 'HKHKG', 'CNSHK', 'GRPIR', 'BRSSZ', 'CNSHK', 'CNSHK', 'SADMM', 'CNYTN', 'GBFXT', 'CNYTN', 'PAONX', 'CNSHK', 'KHKOS', 'CNSHK', 'SIKOP', 'CNHKG', 'RUVVO', 'CNYTN', 'MACAS', 'CNSHK', 'CLSAI', 'CNSHK', 'JPHKT', 'CNYTN', 'ESVAL', 'CNYTN', 'FRLEH', 'CNYTN', 'MATNG', 'CNYTN', 'DZALG', 'CNYTN', 'GRPIR', 'CNSHK', 'TRYAR', 'CNSHK', 'ESVAL', 'CNYTN', 'MTMLA', 'CNSHK', 'LKCMB', 'CNSHK', 'BZBZE', 'CNSHK', 'PKQCT', 'CNYTN', 'SGSIN', 'CNSHK', 'SIKOP', 'CNSHK', 'ITGOA', 'CNSHK', 'ZADUR', 'CNYTN', 'CVRAI', 'CNYTN', 'MYTPP', 'CNSHA', 'SGSIN', 'CNSHK', 'OMSOH', 'CNNSA', 'CMKBI', 'HKHKG', 'FRFOS', 'CNYTN', 'GBSOU', 'CNYTN', 'NLRTM', 'CNSHK', 'AEJEA']
# %%由港口名称匹配港口坐标
test_port = list(set(test_port))
port_lonlat = {}
for j in test_port:
    for i in port_df.iterrows():
        if j in i[1]['TRANS_NODE_NAME']:
            port_lonlat[str(transport_trace_dict[j])] = (i[1]['LONGITUDE'], i[1]['LATITUDE'])
            break

test_port = [str(transport_trace_dict[i]) for i in test_port]
test_list = ['CNYTN-VNVUT-SGSIN-FRLEH','CNYTN-HRRIJ','CNSHK-INNSA-LKCMB-AEJEA-AEAUH-KWKWI','CNSHK-CLSAI']

# test_list = ['LTKLJ-HKHKG']
# 将字符串变成数字编码
# test_list = [[str(transport_trace_dict[i.split('-')[0]]), str(transport_trace_dict[i.split('-')[-1]])] for i in
#              test_list]
test_list = [list(map(lambda x:str(transport_trace_dict[x]), trace.split('-'))) for trace in test_list]

new_test_list = []
for i in test_list:
    if i not in new_test_list:
        new_test_list.append(i)
    
test_list = new_test_list


# In[8]:


new_col = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
           'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
           'vesselStatus', 'TRANSPORT_TRACE', 'distance']

# %% 开始截取历史路径
print('开始读取文件')
train_dir = './data/all_train_data_zip.csv'
train_data = pd.read_csv(train_dir,
                         usecols=new_col)
# test_data = pd.read_csv('./data/R2 ATest 0711.csv')
test_data = pd.read_csv('./data/Btest0711_ALL.csv')
print('读取文件完成')
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], infer_datetime_format=True)
train_data.sort_values(['loadingOrder', 'timestamp'], inplace=True)
group_df = train_data.groupby('loadingOrder')


# In[11]:


df_list = []
len_df = len(group_df)
cnt = 0
lishi_num = 0
long_trace = []
for order_name, df in group_df:
    cnt += 1

    # 去除只有一个GPS数据的路径
    if len(df) == 1:
        print('*'*40)
        continue
    if df.empty:
        print('*'*40)
        continue
    # 获取当前订单的路由
    TRANSPORT_TRACE = ''
    for i in df['TRANSPORT_TRACE'].values:
        if str(i) != '-1':
            TRANSPORT_TRACE = str(i)
    if TRANSPORT_TRACE == '':
        continue
    old_trace = TRANSPORT_TRACE
    # 将路由变为列表
    if '-' in TRANSPORT_TRACE:
        TRANSPORT_TRACE = TRANSPORT_TRACE.split('-')
    else:
        continue

    # %% 开始判断是否为所需的历史路由
#     print('开始判断是否为所需的历史路由')
    # 判断是否要进行后面的坐标匹配
    do_later = False
    trace_list = []
    # 测试集路由与当前订单的路由进行匹配，匹配成功就开始截取历史路径
    for i in test_list:
        if i[-1] in TRANSPORT_TRACE:# and i[-1] in TRANSPORT_TRACE:
            trace_list.append(i)
            do_later = True
    # %% 判断是否要截取路径
    if do_later:
        for trace in trace_list:
            # 采用KD树对训练集订单的坐标进行储存
            lon_lat = np.array([(i, j) for i, j in zip(df['longitude'], df['latitude'])])
            tree = KDTree(lon_lat, leaf_size=2)

            # KD树查询起点、终点港口，提高到岗时间的精准度
            dist_begin, index_begin = tree.query([port_lonlat[trace[0]]], k=1)
            dist_end, index_end = tree.query([port_lonlat[trace[-1]]], k=1)
            index_begin, index_end = index_begin[0][0],index_end[0][0] 

            df_temp = df.iloc[index_begin:index_end + 1].copy()
            df_temp['TRANSPORT_TRACE'] = old_trace
            df_temp['begin'] = trace[0]
            df_temp['end'] = trace[-1]
            df_temp['new_trace'] = '-'.join(trace)
            df_list.append(df_temp)
            lishi_num+=1
            print("第%d个文件" % 0, '第%d订单' % cnt, len_df, lishi_num)


# In[12]:


dff = pd.concat(df_list)

# df = pd.concat(df_list)
dff.info()


# In[14]:


dff.to_csv('./history_trace/B_4_no_history_trace.csv',index = False)

