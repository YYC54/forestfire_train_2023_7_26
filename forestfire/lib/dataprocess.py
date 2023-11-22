import sys
sys.path.append(".")
import glob
import pandas as pd
import numpy as np
import os
from typing import Generator
import re
from lib.untils import untils,interp,zh,ca,usa
# from datetime import datetime
import configparser
from datetime import datetime, timedelta
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import warnings
from pandas.errors import DtypeWarning
from scipy.spatial import cKDTree
from pykrige.ok import OrdinaryKriging
from numpy.linalg import LinAlgError

config = configparser.ConfigParser()
print(sys.path[0])
config.read(os.path.join(sys.path[0], 'configs.ini'))
# config.read('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/configs.ini')

class Dataprocess(object):
    def __init__(self,stime,bound):
        # self.stime = datetime.strptime(stime, '%Y%m%d')
        # self.etime = datetime.strptime(etime, '%Y%m%d')
        self.stime = stime
        self.bound = bound
        self.ne_obs = None #东北气象要素数据
        self.sw_obs = None # 西南气象要素数据
        self.zh_ne_data = None # 东北ffdi数据
        self.zh_sw_data = None  # 西南ffdi数据
        self.ca_ne_data = None  # 东北canada数据
        self.ca_sw_data = None  # 西南canada数据
        self.us_ne_data = None  # 东北usa数据
        self.us_sw_data = None  # 西南usa数据
        self.merge_sw = None
        self.merge_ne = None
        self.obs = config.get("BASE_PATH","obs") #站点原始数据
        self.zh_data = config.get("BASE_PATH","zh_model")#zh数据路径
        self.ca_data = config.get("BASE_PATH","ca_model")
        self.us_data = config.get("BASE_PATH","us_model")
    def obs_data(self):
        stime = datetime.strptime(self.stime, '%Y%m%d')
        elements = ['tmin', 'tmax', 'rh', 'wind', 'rr24']
        # ,'wind','rr24''/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/read_micaps/预报样例数据/气象要素'
        ne_df = pd.DataFrame()
        sw_df = pd.DataFrame()
        for element in elements:
            ne, sw = untils.obs(
                self.obs,
                stime, element, self.bound)
            # 东北
            if ne_df.empty:
                ne_df = ne
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                ne_df = ne_df.merge(ne, on=['lon', 'lat', 'time'], how='outer')

            # 西南
            if sw_df.empty:
                sw_df = sw
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                sw_df = sw_df.merge(sw, on=['lon', 'lat', 'time'], how='outer')

        columns = { 'tmax':"TEM_Max", 'tmin':"TEM_Min",
                    'rh':"RHU_Min",
                           'rr24':"PRE_Time_2020", 'wind':"WIN_S_Max"}

        ne_df.rename(columns=columns,inplace=True)
        sw_df.rename(columns=columns, inplace=True)
        # 东北
        self.ne_obs = ne_df
        # 西南
        self.sw_obs = sw_df
        print("气象要素处理完毕")


    def model_data(self):
        # time = '20231113''/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/read_micaps/预报样例数据/中国'
        stime = datetime.strptime(self.stime, '%Y%m%d')
        elements = ['ffdi']
        ne_df = pd.DataFrame()
        sw_df = pd.DataFrame()
        for element in elements:
            ne, sw = zh.model_ch(self.zh_data, stime, element, self.bound)
            # 东北
            if ne_df.empty:
                ne_df = ne
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                ne_df = ne_df.merge(ne, on=['lon', 'lat', 'time'], how='outer')

            # 西南
            if sw_df.empty:
                sw_df = sw
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                sw_df = sw_df.merge(sw, on=['lon', 'lat', 'time'], how='outer')
        columns = {'ffdi':'FFDI'}
        ne_df.rename(columns=columns, inplace=True)
        sw_df.rename(columns=columns, inplace=True)
        self.zh_ne_data = ne_df
        self.zh_sw_data = sw_df

        elements = ['KBDI', 'ERC', 'SC', 'BI', 'IC', 'P', 'yth1', 'yth10', 'yth100', 'yth1000']
        # ,'wind','rr24'
        ne_df = pd.DataFrame()
        sw_df = pd.DataFrame()
        for element in elements:
            ne, sw = usa.model_usa(
                self.us_data, stime,
                element, self.bound)
            # 东北
            if ne_df.empty:
                ne_df = ne
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                ne_df = ne_df.merge(ne, on=['lon', 'lat', 'time'], how='outer')

            # 西南
            if sw_df.empty:
                sw_df = sw
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                sw_df = sw_df.merge(sw, on=['lon', 'lat', 'time'], how='outer')
        columns = {'KBDI':'kb','ERC':'erc','SC':'sc','BI':'bi','IC':'ic','P':'p'}

        ne_df.rename(columns=columns, inplace=True)
        sw_df.rename(columns=columns, inplace=True)
        self.us_ne_data = ne_df
        self.us_sw_data = sw_df

        elements = ['FFMC', 'DMC', 'DC', 'FWI', 'ISI', 'BUI', 'DSR']
        # ,'wind','rr24'
        ne_df = pd.DataFrame()
        sw_df = pd.DataFrame()
        for element in elements:
            ne, sw = ca.model_ca(
                self.ca_data,
                stime, element, self.bound)
            # 东北
            if ne_df.empty:
                ne_df = ne
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                ne_df = ne_df.merge(ne, on=['lon', 'lat', 'time'], how='outer')

            # 西南
            if sw_df.empty:
                sw_df = sw
            else:
                # 合并数据，这里假设 '经度', '纬度', '时间' 是合并的关键字
                sw_df = sw_df.merge(sw, on=['lon', 'lat', 'time'], how='outer')
        self.ca_ne_data = ne_df
        self.ca_sw_data = sw_df
        print('模型数据处理完毕')
      
    def merge_data(self):

        self.merge_ne = self.ne_obs.merge(self.zh_ne_data,on=['time','lat','lon'],how='outer')
        self.merge_ne = self.merge_ne.merge(self.us_ne_data,on=['time','lat','lon'],how='outer')
        self.merge_ne = self.merge_ne.merge(self.ca_ne_data,on=['time','lat','lon'],how='outer')

        self.merge_sw = self.sw_obs.merge(self.zh_sw_data,on=['time','lat','lon'],how='outer')
        self.merge_sw = self.merge_sw.merge(self.us_sw_data, on=['time', 'lat', 'lon'], how='outer')
        self.merge_sw = self.merge_sw.merge(self.ca_sw_data, on=['time', 'lat', 'lon'], how='outer')


        print('finish merge')
        return self.merge_ne,self.merge_sw

# if __name__ == "__main__":
#     stime = str(20150408)
#     etime = str(20150709)
#     area = ['辽宁省', '黑龙江省', '吉林省']
#     datapprocess = Dataprocess(stime=stime, etime=etime,area =area)
#     obs_data = datapprocess.obs_data()
#     model_data = datapprocess.model_data()
#     merge_data = datapprocess.merge_data()
#     interp_data = datapprocess.interp()
                