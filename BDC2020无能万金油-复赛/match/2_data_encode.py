#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import json
import numpy as np

"""
本代码为对原始数据的编码（字符串转化为数字），和对数据的降采样
"""


# In[2]:


def get_data(data, mode='train'):
    """
    套用baseline的数据类型转换函数
    :param data:
    :param mode:
    :return:
    """
    assert mode == 'train' or mode == 'test'

    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)
    data['longitude'] = data['longitude'].astype(float)
    data['latitude'] = data['latitude'].astype(float)

    data['direction'].fillna(0, inplace=True)
    data['direction'] = data['direction'].astype(int)

    data['speed'] = data['speed'].astype(float)
    had_proces = ['timestamp', 'longitude', 'latitude', 'speed', 'direction', 'onboardDate', 'vesselNextportETA']
    for col in data:
        if col not in had_proces:
            data[col] = data[col].astype(str)
    return data


# In[3]:



def drop_duplicates(df):
    """
    去除重复值、添加一点属性
    :param df:
    :return:
    """
    '''6.25早上,修改了降采样，和代码顺序（降采样提前），保留时间差值，修改速度异常值'''
    df.sort_values(['loadingOrder', 'timestamp'], inplace=True)

    group_df = df.groupby('loadingOrder')

    # %% 按订单分组去除时间重复值
    df = group_df.apply(lambda x: x.drop_duplicates(subset=['timestamp'], keep='first'))
    df.index = range(len(df))

    # %% 对相邻两个坐标时间差小的进行去除
    print('开始降采样')

    '''
    差值小于60的，以步长为2去掉，即去掉一半差值为60的，防止部分路径连续密集化
    差值小于100的，以步长为2去掉
    差值小于180的，以步长为2去掉    
    差值小于220的，以步长为2去掉
    差值小于230的，全去掉
    '''
    df['diff_seconds'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['diff_seconds'].fillna(0, inplace=True)
    df['diff_seconds'] = df['diff_seconds'].astype(int)
    temp_index = df.ix[:, 'diff_seconds'] > 60
    temp_index[temp_index[temp_index == False].index[::2]] = True
    df = df[temp_index]

    df['diff_seconds'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['diff_seconds'].fillna(0, inplace=True)
    df['diff_seconds'] = df['diff_seconds'].astype(int)
    temp_index = df.ix[:, 'diff_seconds'] > 100
    temp_index[temp_index[temp_index == False].index[::2]] = True
    df = df[temp_index]

    df['diff_seconds'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['diff_seconds'].fillna(0, inplace=True)
    df['diff_seconds'] = df['diff_seconds'].astype(int)
    temp_index = df.ix[:, 'diff_seconds'] > 180
    temp_index[temp_index[temp_index == False].index[::2]] = True
    df = df[temp_index]

    df['diff_seconds'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['diff_seconds'].fillna(0, inplace=True)
    df['diff_seconds'] = df['diff_seconds'].astype(int)
    temp_index = df.ix[:, 'diff_seconds'] > 220
    temp_index[temp_index[temp_index == False].index[::2]] = True
    df = df[temp_index]

    df['diff_seconds'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['diff_seconds'].fillna(0, inplace=True)
    df['diff_seconds'] = df['diff_seconds'].astype(int)
    temp_index = df.ix[:, 'diff_seconds'] > 230
    # temp_index[temp_index[temp_index == False].index[::2]] = True
    df = df[temp_index]
    df = df.drop(['diff_seconds'], 1)

    print('降采样完成')

    # %% 整理距离特征
    df['lat_diff'] = df.groupby('loadingOrder')['latitude'].diff(1)
    df['lon_diff'] = df.groupby('loadingOrder')['longitude'].diff(1)

    print('处理完金纬度差值了')
    df['distance'] = np.sqrt(pow(df['lat_diff'], 2) + pow(df['lon_diff'], 2)) * 1e5
    df = df.drop(['lat_diff', 'lon_diff'], 1)
    df['distance'].fillna(0, inplace=True)
    df['distance'] = df['distance'].astype(int)

    # %% 纠正速度特征
    df['diff_seconds'] = df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
    df['diff_seconds'].fillna(0, inplace=True)
    df['diff_seconds'] = df['diff_seconds'].astype(int)

    df['new_speed'] = df['distance'] / df['diff_seconds'] * 3.6
    df['new_speed'].fillna(0, inplace=True)
    df['new_speed'] = df['new_speed'].astype(int)

    df['speed'].fillna(0, inplace=True)
    print('开始map速度')
    # df['speed'] = df[['new_speed', 'speed']].apply(lambda x: x['new_speed'] if x['speed'] <= 3.6 else x['speed'], 1)
    # where语法，运行速度变快， 对速度小于3.6 和25的重新计算
    df['speed'] = df['new_speed'].where((df['speed'] <= 3.6) | (df['speed'] > 25), df['speed'])
    print('map速度完成')
    df['speed'] = df['speed'].astype(int)

    df = df.drop(['new_speed'], 1)

    print('处理完重复值了')

    return df


# In[4]:



def transport_trace_map(x):
    """
    对TRANSPORT_TRACE编码
    :param x:
    :return:
    """
    global transport_trace_dict
    s = ''
    for i in x.replace(' ', '').replace('<', '').replace('>', '').split('-'):
        if i != '':
            s += str(transport_trace_dict[i]) + '-'
    return s[:-1]


def next_port_replace_by_trace(x, trace_list):
    """
    检查vesselNextport是否在trace里面
    :param x:
    :param trace_list:
    :return:
    """
    for i in trace_list[::-1]:
        if i in x:
            return i
    return 'nan'


# In[5]:



# %%制作映射字典
loadingOrder_list = json.load(open('./data/loadingOrder_list.json', 'r'))
loading_order_dict = {j: i for i, j in enumerate(sorted(loadingOrder_list))}

vessel_status = ['under way using engine', 'at anchor', 'moored', 'not under command', 'under way sailing',
                 'constrained by her draught']
vessel_status_dict = {}
for index, i in enumerate(vessel_status):
    vessel_status_dict[i] = index + 1

vessel_mmsi_dict = {'vessel_frist_name': 0}  # f:i for i,f in enumerate(train_df.TRANSPORT_TRACE.unique())
carrier_name_dict = {'carrier_frist_name': 0}
next_port_dict = {'nan': 0}
transport_trace_dict = {'nan': -1}

# %%制作usecol'''
file_path = './data/%d.csv'
columns = ['loadingOrder', 'carrierName', 'timestamp', 'longitude',
           'latitude', 'vesselMMSI', 'speed', 'direction', 'vesselNextport',
           'vesselNextportETA', 'vesselStatus', 'vesselDatasource', 'TRANSPORT_TRACE']
# 去掉两列数据
drop = ['vesselNextportETA', 'vesselDatasource']
# 将列名改为索引
use_col = []
for i, col in enumerate(columns):
    if col not in drop:
        use_col.append(i)


# In[6]:



# %% batch处理文件'''
all_train_data_zip_list = []
flag = True
for file_num in range(15):
    # file_num = 0
    print('开始读取文件')
    train_data = pd.read_csv(file_path % file_num, header=None, usecols=use_col)
    print('读取文件完成')
    train_data.columns = [j for i, j in enumerate(columns) if i in use_col]
    train_data = get_data(train_data, mode='train')
    print('开始map订单号')
    train_data['loadingOrder'] = train_data['loadingOrder'].apply(lambda x: loading_order_dict[x])
    print('map订单号完成')

    # %% 降采样、去重复值
    train_data = drop_duplicates(train_data)

    print('开始apply船状态')
    train_data['vesselStatus'] = train_data['vesselStatus'].apply(lambda x: vessel_status_dict.get(x, 7))
    print('apply船状态完成')

    # %% 对船和公司进行编号
    vessel_mmsi_max = max(vessel_mmsi_dict.values()) + 1
    for i in train_data['vesselMMSI'].unique():
        if i not in vessel_mmsi_dict:
            vessel_mmsi_dict[i] = vessel_mmsi_max
            vessel_mmsi_max += 1

    carrier_name_max = max(carrier_name_dict.values()) + 1
    for i in train_data['carrierName'].unique():
        if i not in carrier_name_dict:
            carrier_name_dict[i] = carrier_name_max
            carrier_name_max += 1
    # %%对路由和下一站的名称进行（去掉一些杂乱的符号等）
    train_data['TRANSPORT_TRACE'] = train_data['TRANSPORT_TRACE'].apply(
        lambda x: str(x).replace(' ', '').replace('<', '').replace('>', ''))
    train_data['vesselNextport'] = train_data['vesselNextport'].apply(
        lambda x: str(x).replace('!', '').replace(' ', ''))
    group_df = train_data.groupby('loadingOrder')

    # %% 判断下一站是否在trace路由中，不在则置空值
    df_list = []
    for name, df in group_df:
        trace = ''
        for i in df['TRANSPORT_TRACE']:
            if str(i) != 'nan':
                trace = str(i)
        if trace == '' or trace == 'nan':
            df['vesselNextport'] = ['nan'] * len(df)
            df_list.append(df)
            continue
        trace_list = trace.split('-')
        trace_list = [i for i in trace_list if i != '']
        df['vesselNextport'] = df['vesselNextport'].apply(lambda x: next_port_replace_by_trace(x, trace_list))
        df_list.append(df)
    train_data = pd.concat(df_list)

    # %% 对路由和下一站编码
    transport_trace_max = max(transport_trace_dict.values()) + 1
    for i in train_data['TRANSPORT_TRACE'].unique():
        for j in i.replace(' ', '').replace('<', '').replace('>', '').split('-'):
            if j not in transport_trace_dict:
                transport_trace_dict[j] = transport_trace_max
                transport_trace_max += 1

    next_port_max = max(next_port_dict.values()) + 1
    for i in train_data['vesselNextport'].unique():
        if i not in next_port_dict:
            next_port_dict[i] = next_port_max
            next_port_max += 1

    # %%对船、公司、路由、下一站进行数字化
    print('字典制作完成')
    print('开始apply船id和公司名称')
    train_data['vesselMMSI'] = train_data['vesselMMSI'].apply(lambda x: vessel_mmsi_dict.get(x))
    train_data['carrierName'] = train_data['carrierName'].apply(lambda x: carrier_name_dict.get(x))
    print('apply船id和公司名称完成')
    print('开始匹配nextport和trace')
    train_data['vesselNextport'] = train_data['vesselNextport'].apply(lambda x: next_port_dict[x])
    train_data['TRANSPORT_TRACE'] = train_data['TRANSPORT_TRACE'].apply(
        lambda x: transport_trace_map(x))
    print('匹配nextport和trace完成')

    all_train_data_zip_list.append(train_data)

    train_data.to_csv('./data/new_%d.csv' % file_num, index=False)


# In[7]:


# %% 保存编码字典 和 采样后的数据集
with open('./data/loading_order_dict.json', 'w') as f:
    json.dump(loading_order_dict, f)

with open('./data/next_port_dict.json', 'w') as f:
    json.dump(next_port_dict, f)

with open('./data/transport_trace_dict.json', 'w') as f:
    json.dump(transport_trace_dict, f)

with open('./data/vessel_mmsi_dict.json', 'w') as f:
    json.dump(vessel_mmsi_dict, f)

with open('./data/vessel_status_dict.json', 'w') as f:
    json.dump(vessel_status_dict, f)

with open('./data/carrier_name_dict.json', 'w') as f:
    json.dump(carrier_name_dict, f)

new_data = pd.concat(all_train_data_zip_list)
new_data.to_csv('./data/all_train_data_zip.csv', index=False)


# In[8]:


new_data.head()

