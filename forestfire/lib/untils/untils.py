import pandas as pd
# import tensorflow as tf
import os
from typing import Generator
import re
import numpy as np
import datetime
from glob import glob
from pathlib import Path
from math import radians, cos, sin, asin, sqrt

def find_txt_files(directory, date):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        # print("Root:", root)
        # print("Dirs:", dirs)
        # print("Files:", files)
        for file in files:
            # print(file)
            if file.endswith('.txt') and str(date) == file.split('.')[0]:
                txt_path = os.path.join(root, file)
                # print(txt_path)
                txt_files.append(txt_path)
                # print("Found:", txt_path)
    # print("All txt files:", txt_files)
    print(txt_files)
    return txt_files


def read_txt_file(file_path):
    # 从文件名中提取年、月、日
    # date_str = os.path.basename(file_path).split('.')[0]  # 例如，得到"20100101"
    # year = int(date_str[:4])
    # month = int(date_str[4:6])
    # day = int(date_str[6:])

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # # 将年、月、日添加为新列
    # df['year'] = year
    # df['month'] = month
    # df['day'] = day

    return df

def txt_file_generator(file_list: list) -> Generator[pd.DataFrame, None, None]:
    for txt_file in file_list:
        yield read_txt_file1(txt_file)


def read_txt_file1(file_path):
    # 从文件名中提取年、月、日
    date_str = os.path.basename(file_path).split('.')[0]  # 例如，得到"20100101"
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:])

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # 将年、月、日添加为新列
    df['year'] = year
    df['month'] = month
    df['day'] = day

    return df



def read_micaps4(file_path):
    '''

    :param file_path:
    :return:
    '''
    with open(file_path) as f:
        lines = f.readlines()

    header = lines[:2]
    # 使用空格分隔并过滤掉空字符串
    header_values = list(filter(None, header[1].split(' ')))  #返回list
    header_values = [float(x) if '.' in x else int(x) for x in header_values]

    # data = [list(map(float, line.split())) for line in lines[2:]]   #返回list
    data = [list(map(float, line.replace(',', ' ').split())) for line in lines[2:]]

    return header, header_values, pd.DataFrame(data)
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
def get_path(path,time,element,bound):
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
    if element == 'rr24':
        order = 24
    else:
        order = 6
    for _ in range(bound):  # 循环10次
        order_str = f'{order:03}'  # 将bound转换为三位数的字符串格式
        print(f'{os.path.join(path)}/{element}/{YYYYmd}08.{order_str}')
        file_list = glob(
            f'{os.path.join(path)}/{element}/{YYYYmd}08.{order_str}')
        # print(file_list)
        # order = int(order)  # 确保 order 是整数
        order += 24  # 每次增加24
        files.extend(file_list)

    return files
def obs(path,time, element, bound):
    '''

    :param file_path:
    :return:
    '''
    files = get_path(path, time, element, bound)
    data_ne_all = []
    data_sw_all = []
    for file_path in files:
        file_name = file_path.split('/')[-1]
        file_time = file_name.split('.')[0][0:8]
        stime = datetime.datetime.strptime(file_time, '%Y%m%d')
        if element == 'rr24':
            order = (int(file_name.split('.')[-1]
                         ) - 24) / 24
        else:
            order = (int(file_name.split('.')[-1]
                    )-6)/24
        # order = int(bound)
        time_order = datetime.timedelta(days=order)
        # 计算新日期
        time = stime + time_order
        print(f'正在处理的{element}文件时间：',time)
        header, header_values, data = read_micaps4(file_path)
        data_ne , data_sw = micaps4_lonlat(data,element ,time)

        data_ne_all.append(data_ne)
        data_sw_all.append(data_sw)


    final_data_ne = pd.concat(data_ne_all, ignore_index=True)
    final_data_sw = pd.concat(data_sw_all, ignore_index=True)
    return final_data_ne,final_data_sw