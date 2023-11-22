import datetime
import os
from glob import glob
import pandas as pd
from pathlib import Path
import numpy as np
from math import radians, cos, sin, asin, sqrt
import chardet
def read_micaps4(file_path):
    '''

    :param file_path:
    :return:
    '''
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    encoding = result['encoding']


    with open(file_path,encoding=encoding) as f:
        lines = f.readlines()

    header = lines[:2]
    # 使用空格分隔并过滤掉空字符串
    header_values = list(filter(None, header[1].split(' ')))  #返回list
    header_values = [float(x) if '.' in x else int(x) for x in header_values]

    # data = [list(map(float, line.split())) for line in lines[2:]]   #返回list
    data = [list(map(float, line.replace(',', ' ').split())) for line in lines[2:]]
    data = pd.DataFrame(data)
    # 将 DataFrame 转换为一维数组
    data_flat = data.values.flatten()

    # 移除 NaN 值
    data_flat = data_flat[~np.isnan(data_flat)]

    # 计算每行的列数
    columns_per_row = len(data_flat) // 1201

    # 重塑数组为 1201 行
    data_reshaped = data_flat.reshape(1201, columns_per_row)

    # 创建新的 DataFrame
    data = pd.DataFrame(data_reshaped)


    return header, header_values, data
def micaps4_lonlat(data,element,time):
    '''
    将横竖转化为经纬度
    :param data:
    :return:
    '''

    lon_min, lon_max, lat_min, lat_max = 70.0, 140.0, 0.0, 60.0
    n_lon, n_lat = 1401, 1201

    # 计算步长
    lon_step = (lon_max - lon_min) / (n_lon - 1)
    lat_step = (lat_max - lat_min) / (n_lat - 1)

    # 生成经纬度坐标
    lons = np.linspace(lon_min, lon_max, n_lon)
    lats = np.linspace(lat_min, lat_max, n_lat)

    # 假设data是您的数据
    # data = pd.DataFrame(np.random.rand(n_lat, n_lon))  # 使用随机数据作为示例

    # 映射数据到经纬度
    data.columns = lons
    data.index = lats

    # 现在data的行列名对应于经纬度
    list_data = []
    for lat, row in data.iterrows():
        for lon, value in row.items():  # 使用 items() 替代 iteritems()
            list_data.append([lon, lat, value])
    data = pd.DataFrame(list_data,columns=['lon','lat',element])
    data['time'] = time

    # 定义东北三省的经纬度范围
    lon_min_ne, lon_max_ne = 120.0, 135.0
    lat_min_ne, lat_max_ne = 40.0, 53.0

    # 筛选出在这个范围内的数据
    data_ne = data[(data['lon'] >= lon_min_ne) & (data['lon'] <= lon_max_ne) &
                   (data['lat'] >= lat_min_ne) & (data['lat'] <= lat_max_ne)]

    # 定义西南地区（四川、贵州、重庆、云南）的经纬度范围
    lon_min_sw, lon_max_sw = 97.0, 110.0
    lat_min_sw, lat_max_sw = 21.0, 35.0

    # 筛选出在这个范围内的数据
    data_sw = data[(data['lon'] >= lon_min_sw) & (data['lon'] <= lon_max_sw) &
                   (data['lat'] >= lat_min_sw) & (data['lat'] <= lat_max_sw)]

    return data_ne , data_sw
def get_path_usa(path,time,element,bound):
    '''
    寻找文件，并且获取该文件的预报时间
    :param path:
    :param time:
    :param element:
    :param bound:
    :return:
    '''
    Y = str(time.year)
    Ym = str(time.year) + str(time.month).zfill(2)
    YYYYmd = str(time.year) + str(time.month).zfill(2) + str(time.day).zfill(2)
    mdH = str(time.month).zfill(2) + str(time.day).zfill(2)
    files = []

    order = 24

    for _ in range(bound):  # 循环10次
        order_str = f'{order:03}'  # 将bound转换为三位数的字符串格式
        if element in ['yth1','yth10','yth100','yth1000']:
            print(f'{os.path.join(path)}/yth1-yth10-yth100-yth1000/{element}_{YYYYmd}08.{order_str}')
            file_list = glob(
                f'{os.path.join(path)}/yth1-yth10-yth100-yth1000/{element}_{YYYYmd}08.{order_str}')
        else:
            print(f'{os.path.join(path)}/{element}_{YYYYmd}08.{order_str}')
            file_list = glob(
                f'{os.path.join(path)}/{element}_{YYYYmd}08.{order_str}')
        # print(file_list)
        # order = int(order)  # 确保 order 是整数
        order += 24  # 每次增加24
        files.extend(file_list)

    return files
def model_usa(path,time, element, bound):
    '''

    :param file_path:
    :return:
    '''
    files = get_path_usa(path, time, element, bound)
    data_ne_all = []
    data_sw_all = []
    for file_path in files:
        file_name = file_path.split('/')[-1]
        file_time = file_name.split('.')[0].split('_')[-1][0:8]
        stime = datetime.datetime.strptime(file_time, '%Y%m%d')

        order = (int(file_name.split('.')[-1]
                         ) - 24) / 24

        # order = int(bound)
        time_order = datetime.timedelta(days=order)
        # 计算新日期
        time = stime + time_order
        print(f'正在处理的{element}文件时间：',time)
        header, header_values, data = read_micaps4(file_path)
        # print(data)
        data_ne , data_sw = micaps4_lonlat(data,element ,time)

        data_ne_all.append(data_ne)
        data_sw_all.append(data_sw)


    final_data_ne = pd.concat(data_ne_all, ignore_index=True)
    final_data_sw = pd.concat(data_sw_all, ignore_index=True)
    return final_data_ne,final_data_sw
# time ='20231113'
# stime = datetime.datetime.strptime(time,'%Y%m%d')
# elements = ['KBDI','ERC','SC','BI','IC','P','yth1','yth10','yth100','yth1000']
# # ,'wind','rr24'
# ne_df = pd.DataFrame()
# sw_df = pd.DataFrame()
# for element in elements:
#     ne,sw = model_usa('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/read_micaps/预报样例数据/美国',stime,element,10)
#     #东北
#     if ne_df.empty:
#         ne_df = ne
#     else:
#         # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
#         ne_df = ne_df.merge(ne, on=['lon', 'lat', 'time'], how='outer')
#
#     #西南
#     if sw_df.empty:
#         sw_df = sw
#     else:
#         # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
#         sw_df = sw_df.merge(sw, on=['lon', 'lat', 'time'], how='outer')
#
# print('东北',ne_df)
# print('西南》》》》》》》》》》》》',sw_df)
