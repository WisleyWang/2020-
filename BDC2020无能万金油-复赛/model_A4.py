#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import KFold
import lightgbm as lgb
import requests
from sklearn.model_selection import StratifiedKFold
from lightgbm.sklearn import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score,mean_squared_error,explained_variance_score
from scipy.stats import entropy, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import datetime
import gc
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# In[2]:


def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('{:.2f} Mb, {:.2f} Mb ({:.2f} %)'.format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))
    gc.collect()
    return df


# In[3]:


data = pd.read_pickle('./data/sample_180.pkl')
data['timestamp'] = data['timestamp'].astype('str')
data = reduce_mem(data)
gc.collect()


# In[4]:


test = pd.read_csv('./data/Btest0711_ALL.csv')
train1 = pd.read_csv('./data/R2 ATest 0711.csv')
train1_label = base = pd.read_csv('./data/Abase.csv')
port = pd.read_csv('./data/port_2.csv')


# In[5]:


#去掉A榜测试数据中被删掉的评分订单，用于之后加入训练数据进行训练
l1 = ['AC860038925693',
'CS952075060675',
'DM428031991357',
'DS626552529494',
'EI581767201011',
'GA472803281061',
'HL358914564422',
'JE845105704656',
'LK919030439899',
'LR291426429726',
'LY233998601535',
'NJ417242079579',
'PP710466021916',
'PQ602767500334',
'QF723400588858',
'UK663883669352',
'VJ323567531982',
'ZQ798500357614',
'ZS950908209190']
train1 = train1[~train1['loadingOrder'].isin(l1)].reset_index(drop=True)
train1_label = train1_label[~train1_label['loadingOrder'].isin(l1)].reset_index(drop=True)


# In[6]:


#数据去重
data1 = data.drop_duplicates(['loadingOrder','timestamp','vesselMMSI'])
del data


# ### 将所有的文件按照时间顺序进行排序

# In[7]:


data1['timestamp'] = pd.to_datetime(data1['timestamp'], infer_datetime_format=True)
data1 = data1.groupby(['loadingOrder','vesselMMSI']).apply(lambda x: x.sort_values('timestamp')).reset_index(drop=True)


# In[8]:


train1['timestamp'] = pd.to_datetime(train1['timestamp'], infer_datetime_format=True)
train1 = train1.groupby(['loadingOrder','vesselMMSI']).apply(lambda x: x.sort_values('timestamp')).reset_index(drop=True)


# In[9]:


test['timestamp'] = pd.to_datetime(test['timestamp'], infer_datetime_format=True)
test = test.groupby(['loadingOrder','vesselMMSI']).apply(lambda x: x.sort_values('timestamp')).reset_index(drop=True)


# In[10]:


#数据去重
data1 = data1.drop_duplicates(['longitude','vesselMMSI','latitude','loadingOrder'])
test = test.drop_duplicates(['longitude','vesselMMSI','latitude','loadingOrder'])
train1 = train1.drop_duplicates(['longitude','vesselMMSI','latitude','loadingOrder'])


# ### 数据清洗

# In[11]:


def get_sample_anchor(df):
    # 转化为360度数
    df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)   
    tmp=df.groupby(['loadingOrder','vesselMMSI'])
    df['lat_diff'] = tmp['latitude'].diff(1)
    df['lon_diff'] = tmp['longitude'].diff(1)
    df['diff_seconds'] = tmp['timestamp'].diff(1).dt.total_seconds()
    #df['change_ratio'] = (abs(df['lat_diff'])+abs(df['lon_diff']))/((df['diff_seconds'])/60)
    return df
data1 = get_sample_anchor(data1)
test = get_sample_anchor(test)
train1 = get_sample_anchor(train1)
gc.collect()


# In[12]:


#test = test[test['diff_seconds']>=30]
#去掉训练数据中出现两次及以上数据大量偏移的数据
l1 = []
for i,v in data1[((abs(data1['lon_diff'])+abs(data1['lat_diff']))>20)&(abs(data1['diff_seconds'])<86400)].loadingOrder.value_counts().items():
    if v>1:
        l1.append(i)
data1 = data1[~(data1['loadingOrder'].isin(l1))]
del l1
'''#同理去除停港时间很长的数据
l1 = list(data1[(data1['diff_seconds']>864000)&((abs(data1['lon_diff'])+abs(data1['lat_diff']))<1)].loadingOrder.value_counts().index)
data1 = data1[~(data1['loadingOrder'].isin(l1))]
del l1'''
gc.collect()


# In[13]:


#去除direction为-1，和时间差为0的数据，速度小于0或大于等于40的也删除
def get_train_sample(df):
    #df = df.loc[df['direction'] != -1]
    df = df.loc[df['diff_seconds'] != 0]
    df = df.loc[(df['speed']>=0)]
    #df = df.loc[(df['speed']>=0)&(df['speed']<=50)]
    del df['lat_diff'],df['lon_diff'],df['diff_seconds']
    return df
def get_test_sample(df):
    #df = df.loc[df['direction'] != -1]
    df = df.loc[df['diff_seconds'] != 0]
    df = df.loc[(df['speed']>=0)]
    #df = df.loc[(df['speed']>=0)&(df['speed']<=50)]
    del df['lat_diff'],df['lon_diff'],df['diff_seconds']
    return df
data1 = get_train_sample(data1)
test = get_test_sample(test)
train1 = get_test_sample(train1)
gc.collect()


# In[14]:


train1['timestamp'] = train1['timestamp'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
train1['timestamp'] = pd.to_datetime(train1['timestamp'], infer_datetime_format=True)


# ### 经纬度特征获取与清洗

# In[15]:


data1 = data1.loc[data1['TRANSPORT_TRACE'].notnull()]
data1['len'] = data1['TRANSPORT_TRACE'].str.split('-')
data1['len'] = data1['len'].str.len()
data1 = data1.loc[(data1['len']>=2)]
gc.collect()


# In[16]:


test['len'] = test['TRANSPORT_TRACE'].str.split('-')
test['len'] = test['len'].str.len()
gc.collect()


# In[17]:


train1['len'] = train1['TRANSPORT_TRACE'].str.split('-')
train1['len'] = train1['len'].str.len()
gc.collect()


# In[18]:


#获取起始点和终点岗口
def get_pot(df):
    df['start_pot'] = df['TRANSPORT_TRACE'].str.split('-').apply(lambda x:x[0])
    df['end_pot'] = df['TRANSPORT_TRACE'].str.split('-').apply(lambda x:x[-1])
    return df
train = get_pot(data1)
train['timestamp'] = train['timestamp'].astype('str')
train = reduce_mem(train)
test = get_pot(test)
train1 = get_pot(train1)
del data1
gc.collect()


# ### 将起点和终点港口多名称的进行统一，并选取与test相同起始和终点的数据

# In[19]:


hk=['HONGKONG','CNHKG','HKHKG','HKG','HONG KONG_HONG KONG','CNSHK']
sz=['CNDCB','CNNSA','YANTIAN','SZX','SHEKOU','CNYTN','YTN','CNCWN','CNSHK','DEHAM','HON']
fos=['FRFOS','FOS']
bey=['LBBEY','BEY']
tnc=['ESALG','TNTUN','MAPTM']


# In[20]:


train.loc[train.end_pot.isin(hk),'end_pot']='CNHKG'
train.loc[train.end_pot.isin(sz),'end_pot']='CNYTN'
train.loc[train.end_pot.isin(fos),'end_pot']='FOS'
train.loc[train.end_pot.isin(tnc),'end_pot']=tnc[0]
gc.collect()


# In[21]:


test.loc[test.end_pot.isin(hk),'end_pot']='CNHKG'
test.loc[test.end_pot.isin(sz),'end_pot']='CNYTN'
test.loc[test.end_pot.isin(fos),'end_pot']='FOS'
test.loc[test.end_pot.isin(tnc),'end_pot']=tnc[0]
gc.collect()


# In[22]:


train1.loc[train1.end_pot.isin(hk),'end_pot']='CNHKG'
train1.loc[train1.end_pot.isin(sz),'end_pot']='CNYTN'
train1.loc[train1.end_pot.isin(fos),'end_pot']='FOS'
train1.loc[train1.end_pot.isin(tnc),'end_pot']=tnc[0]
gc.collect()


# In[23]:


train['tra'] = train['start_pot'] + '-' + train['end_pot']
test['tra'] = test['start_pot'] + '-' + test['end_pot']
train1['tra'] = train1['start_pot'] + '-' + train1['end_pot']
gc.collect()


# In[24]:


#获取相同路由数据
train = train[train['tra'].isin(list(test['tra'].value_counts().index))]


# ### 获取港口经纬度

# In[25]:


#test添加起始港口和终点港口的坐标
port1 = port[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'start_pot','LONGITUDE':
                            'start_long','LATITUDE':'start_lat'})
port2 = port[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'end_pot','LONGITUDE':
                            'end_long','LATITUDE':'end_lat'})
test = test.merge(port1,on='start_pot',how='left')
test = test.merge(port2,on='end_pot',how='left')
del port1,port2
gc.collect()


# In[26]:


#test添加起始港口和终点港口的坐标
port1 = port[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'start_pot','LONGITUDE':
                            'start_long','LATITUDE':'start_lat'})
port2 = port[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'end_pot','LONGITUDE':
                            'end_long','LATITUDE':'end_lat'})
train1 = train1.merge(port1,on='start_pot',how='left')
train1 = train1.merge(port2,on='end_pot',how='left')
del port1,port2
gc.collect()


# In[27]:


port1 = port[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'start_pot','LONGITUDE':
                            'start_long_1','LATITUDE':'start_lat_1'})
port2 = port[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'end_pot','LONGITUDE':
                            'end_long_1','LATITUDE':'end_lat_1'})
train = train.merge(port1,on='start_pot',how='left')
train = train.merge(port2,on='end_pot',how='left')
del port1,port2
gc.collect()


# In[28]:


#train添加起始港口和终点港口的坐标
tmp=train.drop_duplicates('loadingOrder',keep='last')
tmp = tmp[['loadingOrder','longitude','latitude']].rename(columns={'longitude':
                                            'end_long','latitude':'end_lat'})
tmp1=train.drop_duplicates('loadingOrder',keep='first')
tmp1 = tmp1[['loadingOrder','longitude','latitude']].rename(columns={'longitude':
                                            'start_long','latitude':'start_lat'})
train = train.merge(tmp,on='loadingOrder',how='left')
train = train.merge(tmp1,on='loadingOrder',how='left')
train = reduce_mem(train)
gc.collect()


# In[29]:


train = train.loc[train['start_long'].notnull()]
train = train.loc[train['start_lat'].notnull()]
train = train.loc[train['end_long'].notnull()]
train = train.loc[train['end_lat'].notnull()]
gc.collect()


# In[30]:


train = train.loc[(abs(train['start_long'] - train['start_long_1'])+abs(train['start_lat'] - train['start_lat_1']))<5] 
train = train.loc[(abs(train['end_long'] - train['end_long_1'])+abs(train['end_lat'] - train['end_lat_1']))<5]
del train['start_long_1'],train['start_lat_1'],train['end_long_1'],train['end_lat_1']
train = reduce_mem(train)
gc.collect()


# In[31]:


#将A榜测试数据加入到训练数据中一起构建特征
del train1['onboardDate']
train = pd.concat([train,train1],axis=0).reset_index(drop=True)


# ### 根据数据中的路由特征按照中间港近似得到船的航行路程

# In[32]:


port3 = pd.read_csv('./data/port_3.csv')


# In[33]:


#将数据按照路由长度分开
def get_int(df):
    df['longitude1'] = df['longitude'].astype(int) // 3 * 3
    df['latitude1'] = df['latitude'].astype(int) // 3 * 3
    return df
#两路由的数据每个点的前一个和后一个港就是首尾港
def get_2_trace(df):
    df['pre_pot'] = df['start_pot']
    df['next_pot'] = df['end_pot']
    df['pre_gap'] = 0
    df['next_gap'] = 0
    df['pre_long_gap'] = 0
    df['pre_lat_gap'] = 0
    df['next_long_gap'] = 0
    df['next_lat_gap'] = 0
    return df
train = get_int(train)
test = get_int(test)
train_1 = train[train['len']==2].reset_index(drop=True)
train_2 = train[train['len']>2].reset_index(drop=True)
test_1 = test[test['len']==2].reset_index(drop=True)
test_2 = test[test['len']>2].reset_index(drop=True)
train_1 = get_2_trace(train_1)
test_1 = get_2_trace(test_1)
gc.collect()


# In[34]:


import operator
#获取当前点的前后港口
def get_cur_port(df): 
    data1 = pd.DataFrame()
    df1 = df.drop_duplicates(['TRANSPORT_TRACE','longitude1','latitude1']).reset_index(drop=True)
    for i in df1.index:
        tmp = df1.iloc[i:i+1]
        l1 = list(list(tmp.TRANSPORT_TRACE.value_counts().index).pop().split('-'))
        c = dict()
        d = []
        for j in l1:
            if j not in list(port3['TRANS_NODE_NAME'].value_counts().index):
                l1.remove(j)
            l2 = l1
        if len(l2)<2:
            continue
        for k in range(len(l2)):
            c[k] = (abs(port3[port3['TRANS_NODE_NAME']==l2[k]].LONGITUDE.values-tmp.longitude.values)+
                abs(port3[port3['TRANS_NODE_NAME']==l2[k]].LATITUDE.values - tmp.latitude.values))
            C = sorted(c.items(),key=operator.itemgetter(1))
        a = C[0][0]
        b = C[1][0]
        if a > b:
            a,b = b,a
        tmp['pre_pot'] = l2[a]
        tmp['next_pot'] = l2[b]
        s1 = ''
        s2 = ''
        for i in l2[:a+1]:
            s1 += '-'+i
        for i in l2[b:]:
            s2 += '-'+i
        tmp['pre_trace'] = s1[1:]
        tmp['next_trace'] = s2[1:]
        data1 = pd.concat([data1,tmp],axis=0)
        del tmp
    return data1
test2 = get_cur_port(test_2)
test2 = test2[['TRANSPORT_TRACE','longitude1','latitude1','pre_pot','next_pot','pre_trace','next_trace']]
test_2 = test_2.merge(test2,on=['TRANSPORT_TRACE','longitude1','latitude1'],how='left')
train2 = get_cur_port(train_2)
train2 = train2[['TRANSPORT_TRACE','longitude1','latitude1','pre_pot','next_pot','pre_trace','next_trace']]
train_2 = train_2.merge(train2,on=['TRANSPORT_TRACE','longitude1','latitude1'],how='left')
gc.collect()


# In[35]:


#获取前后的距离差值
def get_pre_next_gap(df):
    df1 = df.drop_duplicates(['TRANSPORT_TRACE','pre_pot','next_pot']).reset_index(drop=True)
    merge_gap = pd.DataFrame()
    for i in df1.index:
        tmp = df1.iloc[i:i+1]
        c = d = 0
        d_long = d_lat = 0
        d_dis = 0
        c_long = c_lat = 0
        l1 = list(list(tmp.pre_trace.value_counts().index).pop().split('-'))
        l2 = list(list(tmp.next_trace.value_counts().index).pop().split('-'))
        if len(l1)>1:
            for k in range(len(l1)-1):
                if k+1 <= len(l1)-1:
                    d += (abs(port3[port3['TRANS_NODE_NAME']==l1[k+1]].LONGITUDE.values - port3[port3['TRANS_NODE_NAME']==l1[k]].LONGITUDE.values)+
                    abs(port3[port3['TRANS_NODE_NAME']==l1[k+1]].LATITUDE.values - port3[port3['TRANS_NODE_NAME']==l1[k]].LATITUDE.values))
                    d_long += abs(port3[port3['TRANS_NODE_NAME']==l1[k+1]].LONGITUDE.values - port3[port3['TRANS_NODE_NAME']==l1[k]].LONGITUDE.values)
                    d_lat += abs(port3[port3['TRANS_NODE_NAME']==l1[k+1]].LATITUDE.values - port3[port3['TRANS_NODE_NAME']==l1[k]].LATITUDE.values)
                    #d_dis += distance(port3[port3['TRANS_NODE_NAME']==l1[k]].LATITUDE.values,port3[port3['TRANS_NODE_NAME']==l1[k+1]].LATITUDE.values,
                                    #port3[port3['TRANS_NODE_NAME']==l1[k]].LONGITUDE.values,port3[port3['TRANS_NODE_NAME']==l1[k+1]].LONGITUDE.values)
        else:
            d = 0
            d_long = 0
            d_lat = 0
            #d_dis = 0
        tmp['pre_gap'] = d
        tmp['pre_long_gap'] = d_long
        tmp['pre_lat_gap'] = d_lat
        #tmp['pre_distance'] = d_dis
        if len(l2)>1:
            for k in range(len(l2)-1):
                if k+1 <= len(l2)-1:
                    c += (abs(port3[port3['TRANS_NODE_NAME']==l2[k+1]].LONGITUDE.values - port3[port3['TRANS_NODE_NAME']==l2[k]].LONGITUDE.values)+
                    abs(port3[port3['TRANS_NODE_NAME']==l2[k+1]].LATITUDE.values - port3[port3['TRANS_NODE_NAME']==l2[k]].LATITUDE.values))
                    c_long += abs(port3[port3['TRANS_NODE_NAME']==l2[k+1]].LONGITUDE.values - port3[port3['TRANS_NODE_NAME']==l2[k]].LONGITUDE.values)
                    c_lat += abs(port3[port3['TRANS_NODE_NAME']==l2[k+1]].LATITUDE.values - port3[port3['TRANS_NODE_NAME']==l2[k]].LATITUDE.values)
        else:
            c = 0
            c_long = 0
            c_lat = 0
        tmp['next_gap'] = c
        tmp['next_long_gap'] = c_long
        tmp['next_lat_gap'] = c_lat
        merge_gap = pd.concat([merge_gap,tmp],axis=0)
    return merge_gap
test_merge_gap = get_pre_next_gap(test_2)
test_merge_gap = test_merge_gap[['TRANSPORT_TRACE','pre_pot','next_pot','pre_gap','next_gap','pre_long_gap','pre_lat_gap','next_long_gap','next_lat_gap']]
test_2 = test_2.merge(test_merge_gap,on=['TRANSPORT_TRACE','pre_pot','next_pot'],how='left')
del test_merge_gap
del test_2['pre_trace'],test_2['next_trace']
train_merge_gap = get_pre_next_gap(train_2)
train_merge_gap = train_merge_gap[['TRANSPORT_TRACE','pre_pot','next_pot','pre_gap','next_gap','pre_long_gap','pre_lat_gap','next_long_gap','next_lat_gap']]
train_2 = train_2.merge(train_merge_gap,on=['TRANSPORT_TRACE','pre_pot','next_pot'],how='left')
del train_merge_gap
del train_2['pre_trace'],train_2['next_trace']


# In[36]:


train = pd.concat([train_1,train_2],axis=0).reset_index(drop=True)
test = pd.concat([test_1,test_2],axis=0).reset_index(drop=True)
def get_gap(df):
    #将前后港口的经纬度merge进去，用当前的与其相减得到到前后港的距离
    port1 = port3[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'pre_pot','LONGITUDE':
                                'pre_long','LATITUDE':'pre_lat'})
    port2 = port3[['TRANS_NODE_NAME','LONGITUDE','LATITUDE']].rename(columns={'TRANS_NODE_NAME':'next_pot','LONGITUDE':
                                'next_long','LATITUDE':'next_lat'})
    df = df.merge(port1,on='pre_pot',how='left')
    df = df.merge(port2,on='next_pot',how='left')
    del port1,port2
    df['start_gap'] = (abs(df['longitude']-df['pre_long']) + abs(df['latitude']-df['pre_lat'])) + df['pre_gap']
    df['end_gap'] = (abs(df['longitude']-df['next_long']) + abs(df['latitude']-df['next_lat'])) + df['next_gap']
    #--------------------------------------------------------------------------------------------------------------
    df['start_long_gap'] = abs(df['longitude']-df['pre_long']) + df['pre_long_gap']
    df['start_lat_gap'] = abs(df['latitude']-df['pre_lat']) + df['pre_lat_gap']
    df['end_long_gap'] = abs(df['longitude']-df['next_long']) + df['next_long_gap']
    df['end_lat_gap'] = abs(df['latitude']-df['next_lat']) + df['next_lat_gap']
    #df['have_run_distance'] = distance(df.latitude.values,df.pre_lat.values,df.longitude.values,df.pre_long.values)
    #df['cumsum_distance'] = df['have_run_distance'] + df['pre_distance']
    del df['pre_long'],df['pre_gap'],df['next_long'],df['next_gap'],df['longitude1'],df['latitude1'],df['pre_lat'],df['next_lat']
    del df['pre_long_gap'],df['pre_lat_gap'],df['next_long_gap'],df['next_lat_gap']#,df['have_run_distance'],df['pre_distance']
    return df
train = get_gap(train)
test = get_gap(test)
gc.collect()


# In[37]:


def get_all_trace(df):  
    #尝试根据路由构建总的经纬度的变化值
    dic1 = dict()
    for i in list(df['TRANSPORT_TRACE'].value_counts().index):
        l1 = list(i.split('-'))
        c = 0
        for j in l1:
            if j not in list(port3['TRANS_NODE_NAME'].value_counts().index):
                l1.remove(j)
            l2 = l1
        for k in range(len(l2)-1):
            if k+1 <= len(l2):
                c += (abs(port3[port3['TRANS_NODE_NAME']==l2[k+1]].LONGITUDE.values - port3[port3['TRANS_NODE_NAME']==l2[k]].LONGITUDE.values)+
                abs(port3[port3['TRANS_NODE_NAME']==l2[k+1]].LATITUDE.values - port3[port3['TRANS_NODE_NAME']==l2[k]].LATITUDE.values))
        dic1[i] = c
        del l1,l2
    for i,v in dic1.items():
        if v > 0:
            dic1[i] = float('%.6f'%v)
        else:
            dic1[i] = 0
    return dic1
map_dic_train = get_all_trace(train)
map_dic_test = get_all_trace(test)
train['all_gap']=train['TRANSPORT_TRACE'].map(map_dic_train)
test['all_gap']=test['TRANSPORT_TRACE'].map(map_dic_test)


# ### 特征构建

# In[38]:


def get_data(data, model='train'):
    #转换成时间戳，并且将每个运单按照时间排序
    assert model=='train' or model=='test'
    data.sort_values(['loadingOrder','vesselMMSI','timestamp'],inplace=True)
    if model=='train':
        pass
#         data['vesselNextportETA'] = pd.to_datetime(data['vesselNextportETA'], infer_datetime_format=True) 
    else:
        data['onboardDate'] = pd.to_datetime(data['onboardDate'], infer_datetime_format=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'], infer_datetime_format=True)    
    return data
def get_anchor(df):
    # 转化为360度数
    df['direction']=df['direction'].values/10
    tmp=df.groupby(['loadingOrder','vesselMMSI'])
    df['lat_diff'] = abs(tmp['latitude'].diff(1))
    df['lon_diff'] = abs(tmp['longitude'].diff(1))
    df['speed_diff'] = abs(tmp['speed'].diff(1))
    df['direction_diff']= abs(tmp['direction'].diff(1))
    df['diff_seconds'] = tmp['timestamp'].diff(1).dt.total_seconds()
    ### 这样实际是做了一个采样！！ #可以去除重复的记录
    df['anchor'] =((abs(df['lat_diff'])<= 0.03)&(abs(df['lon_diff']) <= 0.03)&(abs(df['speed_diff']) <= 0.3)).astype('int')
    ###  这里标记下船几乎停止的地方
    df['stop']=((abs(df['lat_diff']) <= 0.03)&(abs(df['lon_diff']) <= 0.03)&(abs(df['speed']) <= 1)).astype('int')
    df['delay']=(abs(df['diff_seconds'])>3000).astype('int')
    #diff特征需要除以时间差距
    df['lat_diff'] = df['lat_diff'] / (df['diff_seconds'] / 3600)
    df['lon_diff'] = df['lon_diff'] / (df['diff_seconds'] / 3600)
    df['speed_diff'] = df['speed_diff'] / (df['diff_seconds'] / 3600)
    df['direction_diff'] = df['direction_diff'] / (df['diff_seconds'] / 3600)
    #记录是否停港
    #df['stop']=((abs(df['lat_diff'])<0.02)&(abs(df['lon_diff'])<0.02)&(abs(df['speed'])<10)).astype('int')
    #df['stop_times']=(df['stop']*df['diff_seconds']).cumsum()//3600
    return df
def distance(LatA,LatB,LonA,LonB):
    EARTH_RADIUS = 6378.137 # 千米
    def rad(d):
        return d * np.pi/ 180.0
    s=0
    radLatA = rad(LatA)
    radLatB = rad(LatB) 
    a = radLatA-radLatB
    b = rad(LonA)-rad(LonB)
    s= 2 * np.arcsin(np.sqrt(np.power(np.sin(a / 2),2)+ np.cos(radLatA) * np.cos(radLatB)*np.power(np.sin(b / 2),2)))
    s=s* EARTH_RADIUS
    #  保留两位小数
    s = np.round(s * 100)/100
    s = s * 1000 # 转换成m
    return s
def get_feature(df,model='train'):
     #计算移动方便后面计算轨迹长度 m
    df['move_leng']=distance(df.latitude.values,df.groupby(['loadingOrder','vesselMMSI'])['latitude'
                ].shift(1).values,df.longitude.values,df.groupby(['loadingOrder','vesselMMSI'])['longitude'].shift(1).values)  
    #计算下之前的累计距离
    df['cumsum_distance'] = df.groupby(['loadingOrder','vesselMMSI'])['move_leng'].expanding().sum().values
    #-----------------------------------------------------------------------------------------------------------------------
    #计算下之前的船已经行驶的累计距离
    #df['cusum_distance'] = distance(df.start_long_gap.values,df.start_lat_gap.values,df.start_lat.values,df.start_lat.values+df.start_long_gap.values)
    
    #-----------
    df['cusum_direction'] = df.groupby(['loadingOrder','vesselMMSI'])['direction'].expanding().mean().values
    #df['cusum_mean_speed'] = df.groupby('loadingOrder')['speed'].expanding().mean().reset_index(drop=True)
    df['cusum_stop'] = df.groupby('loadingOrder')['stop'].cumsum()
    df['cusum_speed']=df.groupby(['loadingOrder','vesselMMSI'])['speed'].rolling(window=5).mean().values
    #------------------------------------------------------
    df['direction_valc']=df['direction_diff']/df['diff_seconds']#
    df['mean_speed'] = df['move_leng']/(df['diff_seconds']+0.01)
    # 瞬时加速度 m/s2
    df['instant_acc']=df['mean_speed']/(df['diff_seconds']+0.01)
    
    #获取船航行经度和维度的行驶比例和总航行占比
    df['end_long_gap_1'] = abs(df['end_long']-df['longitude'])
    df['end_lat_gap_1'] = abs(df['end_lat']-df['latitude'])
    df['start_long_gap_1'] = abs(df['start_long']-df['longitude'])
    df['start_lat_gap_1'] = abs(df['start_lat']-df['latitude'])
    #df['start_long_ratio'] = abs(df['longitude']-df['start_long']) / abs(df['end_long']-df['start_long'])
    #df['start_lat_ratio'] = abs(df['latitude']-df['start_lat']) / abs(df['end_lat']-df['start_lat'])
    #df['end_long_ratio'] = abs(df['longitude']-df['end_long']) / abs(df['end_long']-df['start_long'])
    #df['end_lat_ratio'] = abs(df['latitude']-df['end_lat']) / abs(df['end_lat']-df['start_lat'])
    #获取总差距
    #df['all_start_gap'] = abs(df['start_long_gap']) + abs(df['start_lat_gap'])
    df['all_start_ratio'] = df['start_gap'] / df['all_gap']
    #df['all_end_gap'] = abs(df['long_gap']) + abs(df['lat_gap'])
    df['all_end_ratio'] = 1 - df['all_start_ratio']
    
    #获取年月日等时间特征
    df['year'] = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['hour'] = df['timestamp'].dt.hour
    df['time'] = df['year'].astype(str)+'-'+df['month'].astype(str)+'-'+df['day'].astype(str)
    
    ## 得到最早的时间
    tmp=df.drop_duplicates(['loadingOrder','vesselMMSI'],keep='first').reset_index(drop=True)
    tmp=tmp[['loadingOrder','vesselMMSI','timestamp','direction']]
    tmp.columns=['loadingOrder','vesselMMSI','start_time','start_direction']
    df=df.merge(tmp,on=['loadingOrder','vesselMMSI'],how='left')
    if model == 'train':
        df['have_run_time']=(df['timestamp']-df['start_time']).dt.total_seconds()
    if model == 'test':
        df['timestamp'] = df['timestamp'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
        df['timestamp'] = pd.to_datetime(df['timestamp'], infer_datetime_format=True)
        df['onboardDate'] = pd.to_datetime(df['onboardDate'], infer_datetime_format=True)
        df['have_run_time'] = (df['timestamp'] - df['onboardDate']).dt.total_seconds()
    df['distanc2taget']=distance(df.latitude.values,df.end_lat.values,df.longitude.values,df.end_long.values)/1000
    df['start_time'] = df['start_time'].dt.year.astype(str) + '-' + df['start_time'].dt.month.astype(str) + '-' + df['start_time'].dt.day.astype(str)
    df['cumsum_mean_speed'] = df['cumsum_distance']/(df['have_run_time']+0.01)
    # 瞬时加速度 m/s2
    df['cumsum_instant_acc']=df['cumsum_mean_speed']/(df['have_run_time']+0.01)
    
    #df['bearing'] = bearing_array(df.latitude.values,df.longitude.values,
                                               #df.groupby('loadingOrder')['latitude'].shift(1).values,
                                               #df.groupby('loadingOrder')['longitude'].shift(1).values)
    #df['start_bearing'] = bearing_array(df['start_lat'],df['start_long'],df['latitude'],df['longitude'])
    #df['end_bearing'] = bearing_array(df['latitude'],df['longitude'],df['end_lat'],df['end_long'])
    return df
def get_hot(df):
    df['day_tag']=(df.timestamp.dt.year%100)*10000+df.timestamp.dt.month*100+df.timestamp.dt.day
    df = df.merge(hot,on=['day_tag','end_pot'],how='left')
    return df
def type_encoding(train_data,test_data):
    ### ----对类别进行编码
    for f in ['TRANSPORT_TRACE','carrierName','vesselMMSI','time','start_time']:
        unique_set=set(train_data[f].unique().tolist()+test_data[f].unique().tolist())
        unique_dict={ f:i for i,f in enumerate(unique_set)}
        test_data[f]=test_data[f].map(unique_dict)
        train_data[f]=train_data[f].map(unique_dict)
        
    # 港口名称编码
    unique_set=set(train_data['start_pot'].unique().tolist()+test_data['start_pot'].unique().tolist()
                  +train_data['end_pot'].unique().tolist()+test_data['end_pot'].unique().tolist())
    unique_dict={ f:i for i,f in enumerate(unique_set)}
    for f in ['start_pot','end_pot']:
        test_data[f]=test_data[f].map(unique_dict)
        train_data[f]=train_data[f].map(unique_dict)
    return train_data,test_data
def get_label(df):
    #tmp = df.groupby(['loadingOrder','vesselMMSI'])['timestamp'].agg({'time_max':'max'})
    df['endtime'] = pd.to_datetime(df['endtime'], infer_datetime_format=True)
    #df = df.merge(tmp,on=['loadingOrder','vesselMMSI'],how='left')
    df['label'] = (df['endtime'] - df['timestamp']).dt.total_seconds()//3600
    return df


# In[39]:


train = get_data(train,model='train')
train = get_anchor(train)
train = get_feature(train,model='train')
#train = get_label(train)
gc.collect()


# ### 标签构建

# In[40]:


#由于A训练数据已经有ETA所以需要将两个部分数据分开进行标签的构建
train_B = train[~train['loadingOrder'].isin(list(train1['loadingOrder'].value_counts().index))]
train_A = train[train['loadingOrder'].isin(list(train1['loadingOrder'].value_counts().index))]


# In[41]:


#获取A榜测试集的标签
train1_label = train1_label[['loadingOrder','ETA']].drop_duplicates()
train_A = train_A.merge(train1_label,on='loadingOrder',how='left')
train_A['timestamp'] = pd.to_datetime(train_A['timestamp'], infer_datetime_format=True)
train_A['ETA'] = pd.to_datetime(train_A['ETA'], infer_datetime_format=True) 
train_A['label'] = (train_A['ETA'] - train_A['timestamp']).dt.total_seconds()//3600
del train_A['ETA']


# In[44]:


#获取B榜训练数据的标签
train= train_B
text = train.loc[abs(train['distanc2taget'])<=50]
merge_tabel = text[['loadingOrder','timestamp']].drop_duplicates(['loadingOrder'],keep='first').rename(columns={'timestamp':'endtime'})
train = train.merge(merge_tabel,on=['loadingOrder'],how='left')
del merge_tabel
#train = get_hot(train)
train = get_label(train)
gc.collect()


# In[49]:


train = train[train['label']>=0]
del train['endtime']
train['timestamp'] = train['timestamp'].astype('str')
train = reduce_mem(train)
gc.collect()


# In[50]:


train_A['timestamp'] = train_A['timestamp'].astype('str')
train = pd.concat([train,train_A],axis=0).reset_index(drop=True)
del train_A,train_B
gc.collect()


# In[53]:


#test进行特征构建
test = get_data(test,model='test')
test = get_anchor(test)
test = get_feature(test,model='test')
#test = get_hot(test)
gc.collect()


# In[54]:


#labelencode
test1 = test.copy()
train,test1 = type_encoding(train,test1)
gc.collect()


# ### 数据集构建与模型训练

# In[ ]:


features = [c for c in train.columns if c in['carrierName', 'longitude', 'latitude', 'vesselMMSI', 'speed', 'direction', 'len', 'start_pot', 'end_pot', 'end_long',
 'end_lat', 'start_long', 'start_lat', 'start_gap', 'end_gap', 'start_long_gap', 'start_lat_gap', 'end_long_gap', 'end_lat_gap', 'lat_diff','anchor','delay',
 'lon_diff', 'speed_diff', 'direction_diff', 'diff_seconds', 'cusum_direction', 'cusum_speed', 'direction_valc', 'end_long_gap_1','all_gap','cusum_stop',
 'end_lat_gap_1', 'start_long_gap_1', 'start_lat_gap_1', 'year','month','day', 'start_direction', 'have_run_time','all_start_ratio','all_end_ratio','stop']]#cumsum_stop
print(features)
print(len(features))
gc.collect()


# In[61]:


from sklearn.metrics import mean_squared_error,explained_variance_score
from sklearn.model_selection import KFold
from  lightgbm.sklearn import LGBMRegressor
def mse_score_eval(preds, valid):
    labels = valid.get_label()
    scores = mean_squared_error(y_true=labels, y_pred=preds)
    return 'mse_score', scores, True

def build_model(train_data, test, pred, label, seed=2099, is_shuffle=True):
    train_pred = np.zeros((train_data.shape[0], ))
    test_pred = np.zeros((test.shape[0], ))
    n_splits = 5
    # Kfold
    fold = KFold(n_splits=n_splits, shuffle=is_shuffle, random_state=seed)
    kf_way = fold.split(train_data[pred])
    # params
#     test_x=np.concatenate([test[pred].values,geohash_test],axis=1)
    # train
    for n_fold, (train_idx, valid_idx) in enumerate(kf_way, start=1):
        train_x, train_y = train_data[pred].iloc[train_idx].values, train_data[label].iloc[train_idx]
        valid_x, valid_y = train_data[pred].iloc[valid_idx].values, train_data[label].iloc[valid_idx]
#         geohash_tr_x,geohash_val_x=geohash_train[train_idx],geohash_train[valid_idx]
#         train_x=np.concatenate([train_x,geohash_tr_x],axis=1)
#         valid_x=np.concatenate([valid_x,geohash_val_x],axis=1)
        
        # 数据加载
        clf=LGBMRegressor( learning_rate=0.5,
        n_estimators=6000,
        boosting_type = 'gbdt',
        objective = 'regression',
        num_leaves=156,
        subsample=0.8,
        njobs=-1,
        max_depth=6,
        reg_lambda=0,
        colsample_bytree=0.8,
        random_state=2019,  # 2019
        metric=['mse'])
        
        clf.fit(
        train_x, train_y,
        eval_set=[(valid_x, valid_y)],
        eval_metric=['mse'],
        categorical_feature='auto',
        early_stopping_rounds=100,
        verbose=100)        
        
        train_pred[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration_)
        
        
        test_pred += clf.predict(test[pred], num_iteration=clf.best_iteration_)/fold.n_splits
    
    print('mean_squared_error:',mean_squared_error(train_data[label].values,train_pred))
    test['label'] = test_pred
    return test[['loadingOrder', 'label']],clf


def bulid_onetrain(train_data, test,pred= features,label= 'label',seed=1099,est=6000, is_shuffle=True):
    train_x,train_y=train_data[features].values,train_data[label].values
    clf=LGBMRegressor( learning_rate=0.01,
    boosting_type = 'gbdt',
    objective = 'regression',
    n_estimators=est,
    num_leaves=156,
    subsample=0.8,
    njobs=-1,
    max_depth=8,
    reg_lambda=0,
    colsample_bytree=0.8,
    random_state=2019,  # 2019
    metric=['mse'])

    clf.fit(
    train_x, train_y,
    eval_set=[(train_x, train_y)],
    eval_metric=['mse'],
    categorical_feature='auto',
    verbose=100)        

    #train_pred= clf.predict(train_x, num_iteration=clf.best_iteration_)


    test_pred= clf.predict(test[pred], num_iteration=clf.best_iteration_)

    #print('mean_squared_error:',mean_squared_error(train_y,train_pred))
    test['label'] = test_pred
    return test[['loadingOrder', 'label']],clf
#result,clf = build_model(train1, test1,pred= features,label= 'label', is_shuffle=True)
result,clf=bulid_onetrain(train, test1,pred= features,label= 'label',est=8000,is_shuffle=True)


# ### 根据预测结果获得预测时间

# In[ ]:


test4 = test1.copy()


# In[64]:


#尝试用最后一条的预测时间
test4['onboardDate'] = pd.to_datetime(test4['onboardDate'])
test4['timestamp'] = pd.to_datetime(test4['timestamp'])
test4['timestamp'] = test4['timestamp'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test4['timestamp'] = pd.to_datetime(test4['timestamp'])
test4['ETA']=(test4['timestamp']+test4['label'].apply(lambda x:pd.Timedelta(hours=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
test4 = test4.drop_duplicates('loadingOrder',keep='last')
test4['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
result1 = test4[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]


# In[65]:


result3 = result1[['loadingOrder','ETA']].drop_duplicates('loadingOrder')
test3 = pd.read_csv('./data/Btest0711_ALL.csv')
test3 = test3.merge(result3,on='loadingOrder',how='left')
test3['creatDate'] = pd.datetime.now().strftime('%Y/%m/%d  %H:%M:%S')
result2 = test3[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI', 'onboardDate', 'ETA', 'creatDate']]


# In[71]:


#转换格式
result2['onboardDate'] = pd.to_datetime(result2['onboardDate'])
result2['onboardDate'] = result2['onboardDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
result2['ETA'] = pd.to_datetime(result2['ETA'])
result2['ETA'] = result2['ETA'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))
result2['creatDate'] = pd.to_datetime(result2['creatDate'])
result2['creatDate'] = result2['creatDate'].apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))


# In[ ]:


#保存得到的结果
result2.to_csv('./result/A4.csv')

