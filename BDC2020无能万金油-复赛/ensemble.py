import pandas as pd

path='./result/'
files=['A1','A2','A3','A4','A5','A6']
for i,f in enumerate(files):
	result=pd.read_csv('./data/{}.csv'.format(f))
	result['base_time'] = '2019/01/01  00:00:00'
	result['base_time'] = pd.to_datetime(result['base_time'])
	result['ETA'] = pd.to_datetime(result['ETA'])
	result['time_gap'] = (result['ETA'] - result['base_time']).dt.total_seconds()
	if i==0:
		ensemble=result.copy()
		ensemble['gap'] = 0
	ensemble['gap'] += result['time_gap']

ensemble['gap']/=len(files)
ensemble['ETA2'] = (ensemble['base_time']+ensemble['gap'
].apply(lambda x:pd.Timedelta(seconds=x))).apply(lambda x:x.strftime('%Y/%m/%d  %H:%M:%S'))

#得到结果
ensemble= ensemble[['loadingOrder', 'timestamp', 'longitude', 'latitude', 'carrierName', 'vesselMMSI',
    'onboardDate', 'ETA2', 'creatDate']].rename(columns={'ETA2':'ETA'})
ensemble.to_csv('./result/final_ensemble_result.csv')