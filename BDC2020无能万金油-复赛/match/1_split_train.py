#!/usr/bin/env python
# coding: utf-8


import json

root = 'data'
count = 0
interval = 10
loadingOrder_list = []
temp = []
with open('./data/train0711.csv','r') as f:
    print('打开文件完成')
    for index, line in enumerate(f):
        if line[0] == '"':
            begin = 1
        else:
            begin = 0
        temp_index = line.find(',')
        if line[temp_index - 1] == '"':
            end = temp_index - 1
        else:
            end = temp_index

        loadingOrder = line[begin:end]
        if loadingOrder not in loadingOrder_list:
            loadingOrder_list.append(loadingOrder)
        if index % 1e6 == 0:
            print(loadingOrder)
            print(index)
print(len(loadingOrder_list))

with open('./data/loadingOrder_list.json', 'w') as f:
    json.dump(loadingOrder_list, f)


batch_size = 15
batch = len(loadingOrder_list) // batch_size
batch_list = [loadingOrder_list[batch * i:batch * (i + 1)] for i in range(batch_size)]
batch_list[-1].extend(loadingOrder_list[batch * batch_size:])

csv_map_rows = {}
for index, li in enumerate(batch_list):
    for key in li:
        csv_map_rows[key] = index

# 创建15个CSV文件
file_list = [open('./data/%d.csv' % i, 'w') for i in range(batch_size)]
dict_map_rows = {i: [] for i in loadingOrder_list}


root = './data'
count = 0
interval = 10

temp = []
with open('./data/train0711.csv', 'r') as f:
    for index, line in enumerate(f):
        # 读取第一列的订单号
        temp_index = line.find(',')
        if line[0] == '"':
            loadingOrder = line[1:temp_index - 1]
        else:
            loadingOrder = line[0:temp_index]
        file_list[csv_map_rows[loadingOrder]].write(line)
        if index % 1e6 == 0:
            print(loadingOrder)
            print(index)

for f in file_list:
    f.close()


