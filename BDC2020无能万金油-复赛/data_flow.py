import pandas as pd
import numpy as np
from tqdm import tqdm


names= ['loadingOrder','carrierName','timestamp','longitude',
                  'latitude','vesselMMSI','speed','direction','vesselNextport',
                  'vesselNextportETA','vesselStatus','vesselDatasource','TRANSPORT_TRACE']

datas=pd.read_csv('./data/train0711.csv',chunksize=500000,names=names)
train_df=pd.DataFrame()
for df in tqdm(datas):
    df.drop(['vesselNextport','vesselNextportETA','vesselStatus','vesselDatasource'],axis=1,inplace=True)
    df['direction']=df['direction'].values.astype('int32')
    df['speed']=df['speed'].values.astype('int8')
    df['timestamp']=pd.to_datetime(df.timestamp)
    train_df=train_df.append(df)
    
print(train_df.shape)
train_df=train_df.loc[train_df.TRANSPORT_TRACE.notna()]
print(train_df.shape)

## 排序采样，减少内存
train_df.sort_values(['loadingOrder','timestamp'],inplace=True)
train_df['diff_seconds'] = train_df.groupby('loadingOrder')['timestamp'].diff(1).dt.total_seconds()
print(train_df.shape)
train_df=train_df.loc[~(train_df.diff_seconds<180)]
print(train_df.shape)


train_df.reset_index(drop=True,inplace=True)
train_df.to_pickle('./data/sample_180.pkl')