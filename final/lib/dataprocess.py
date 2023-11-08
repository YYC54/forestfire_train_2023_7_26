import sys
sys.path.append(".")
import glob
import pandas as pd
import numpy as np
import os
from typing import Generator
import re
from lib.untils import untils,interp
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
    def __init__(self,stime,etime,area):
        # self.stime = datetime.strptime(stime, '%Y%m%d')
        # self.etime = datetime.strptime(etime, '%Y%m%d')
        self.stime = stime
        self.etime = etime
        self.area = area
        self.obs = None
        self.model = None
        self.merge = None
        self.obss = config.get("BASE_PATH","obs") #站点原始数据
        self.obs_path = config.get("BASE_PATH","obs_path")#站点数据处理后保存位置
        self.modell = config.get("BASE_PATH","model")#模型数据路径
        self.model_path = config.get("BASE_PATH","model_path")#模型数据处理后结果
        self.fire_path = config.get("BASE_PATH","fire_path")#火点数据
        self.interp_result = config.get("BASE_PATH","interp_result")#插值数据保存结果

    def obs_data(self):
        dates = pd.date_range(start=self.stime, end=self.etime, freq='D').strftime('%Y%m%d')
        print('days:',len(dates))
        # years = list(range(2010, 2018))
        # print(years)
        print(dates)
        all_dfs = []
        for date in dates:
            txt_files = untils.find_txt_files(self.obss, date)

            df_list = []
            for txt_file in txt_files:
                columns = ['sta', 'lat', 'lon', "Alti", "Year", "Mon", "Day", "TEM_Max", "TEM_Min", "RHU_Min",
                           "PRE_Time_2020","Snow_Depth", "WIN_S_Max"]
                df = pd.read_csv(txt_file, sep=',', header=None)  # 使用制表符作为分隔符
                df = df.iloc[:, :13]  # 选择前13列
                df.columns = columns  # 设置列名
                df_list.append(df)

            df_obs = pd.concat(df_list,ignore_index=True)
            all_dfs.append(df_obs)

            # df_obs.columns =['sta','lat','lon',"Alti", "Year", "Mon", "Day", "TEM_Max","TEM_Min","RHU_Min", "PRE_Time_2020", "Snow_Depth", "WIN_S_Max"]
            # df_obs.to_csv(f'{self.obs_path}/obs/obs_{year}.csv')
        final_df = pd.concat(all_dfs, ignore_index=True)
        self.obs = final_df.loc[:, ~final_df.columns.str.startswith('Unnamed')]
        print(self.obs)
        print("finish obs")


    def model_data(self):
        dates = pd.date_range(start=self.stime, end=self.etime, freq='D').strftime('%Y%m%d')
        #/public/home/lihf_hx/yyc/森林火险模型/dataset/model/
        print(dates)
        all_dataframes = []
        for date in dates:
            txt_files = untils.find_txt_files(f"{self.modell}/China",date)

            dataframes = []
            for txt_file in txt_files:
                df = untils.read_txt_file(txt_file)
                dataframes.append(df)
            df_CHINA = pd.concat(dataframes)
            # print(df_CHINA)
            df_CHINA.columns =['sta','lon','lat','year','month','day','FFDI']
            all_dataframes.append(df_CHINA)  # Step 2
        df_all_CHINA = pd.concat(all_dataframes)  # Step 3
        df_all_CHINA.to_csv(f'{self.model_path}/CHINA.csv')
        print('china',df_all_CHINA)



        dates = pd.date_range(start=self.stime, end=self.etime, freq='D').strftime('%Y%m%d')
        all_dataframes1 = []
        for date in dates:
            txt_files = untils.find_txt_files(f"{self.modell}/USA",date)
            df_gen = untils.txt_file_generator(txt_files)
            df_USA = pd.concat(df_gen, axis=0)
            # print(df_USA)
            df_USA.columns =['sta','lon','lat','yth1','yth10','yth100','yth1000','kb','erc','sc','bi','ic','p','year','month','day']
            all_dataframes1.append(df_USA)  # Step 2
        df_all_USA = pd.concat(all_dataframes1)  # Step 3
        df_all_USA.to_csv(f'{self.model_path}/USA.csv')
        print('usa',df_all_USA)

        dates = pd.date_range(start=self.stime, end=self.etime, freq='D').strftime('%Y%m%d')
        all_dataframes2 = []
        for date in dates:
            txt_files = untils.find_txt_files(f"{self.modell}/Canada",date)
            df_gen = untils.txt_file_generator(txt_files)
            df_CANADA = pd.concat(df_gen, axis=0)
            df_CANADA.columns =['sta','lon','lat','FFMC','DMC','DC','FWI','ISI','BUI','DSR','year','month','day']
            all_dataframes2.append(df_CANADA)  # Step 2
        df_all_CANADA = pd.concat(all_dataframes2)  # Step 3
        df_all_CANADA.to_csv(f'{self.model_path}/CANADA.csv')

        # df_USA = pd.read_csv(r'E:\fire_dataset\model\USA.csv', index_col=0)
        # df_CANADA = pd.read_csv(r'E:\fire_dataset\model\CANADA.csv',index_col=0)
        # df_CHINA = pd.read_csv(r'E:\fire_dataset\model\China.csv', index_col=0)
        # df_USA = df_USA.drop_duplicates(subset= ['sta'],keep='first',inplace=False)
        # df_CANADA = df_CANADA.drop_duplicates(subset= ['sta'],keep='first',inplace=False)

        merged_uc = df_all_USA.merge(df_all_CANADA, on=['sta', 'lon', 'lat','year','month','day'])
        self.model =merged_uc.merge(df_all_CHINA, on = ['sta','year','month','day','lat','lon'])
        # merged_all.to_csv(r'/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/model.csv')

      
    def merge_data(self):        
        columns_to_drop = ['lat', 'lon']
        self.obs.drop(columns=columns_to_drop, inplace=True)
        columns_to_rename = {
            'Year': 'year',
            'Mon': 'month',
            'Day': 'day'
        }
        
        # 使用rename方法来重命名列
        self.obs.rename(columns=columns_to_rename, inplace=True)
        
        self.merge = self.obs.merge(self.model,on = ['sta','year','month','day'])
        print('finish merge')
        # self.merge.to_csv(f'{self.obs_path}/dataset.csv', index=None)

    def interp(self):
        # dates = pd.date_range(start=self.stime, end=self.etime, freq='D').strftime('%Y%m%d')
        dates = pd.date_range(start=self.stime, end=self.etime, freq='D')
        print(dates)

        try:
            fire = pd.read_excel(self.fire_path,engine='openpyxl')
        except:
            fire = pd.read_csv(self.fire_path,encoding='GB2312')
        fire['图像日期'] = pd.to_datetime(fire['图像日期'],format='mixed')
        fire = fire[fire['图像日期'].dt.date.isin(dates.date)]
        print(fire)

        # ['辽宁省', '黑龙江省', '吉林省']
        provinces = self.area
        fire = fire[fire['地区'].isin(provinces)].copy()
        fire_path = '/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/fire.csv'
        fire.to_csv(fire_path)
        # ----------------------------可燃物，土壤湿度插值------------------------------
        data_path = '/public/home/duyl_hx/firedata/a卫星反演/a可燃物含水率'
        new_column_name = "可燃物含水率"  # 新增列名
        output_file_path = '/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/result_water.csv'
        interp.interpolate_and_save(fire_path, data_path, new_column_name, output_file_path)

        data_path = '/public/home/duyl_hx/firedata/a卫星反演/c土壤湿度'
        new_column_name = "土壤湿度"  # 新增列名
        output_file_path1 = '/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/result.csv'
        interp.interpolate_and_save(output_file_path, data_path, new_column_name, output_file_path1)

        # final_result = pd.merge(result_water, result_rh, on=['地区', '图像日期', '东经', '北纬'])
        final_result = pd.read_csv(output_file_path1)
        name_mapping = {

            '图像日期': 'time',
            '东经': 'lon',
            '北纬': 'lat'

        }
        final_result.rename(columns=name_mapping, inplace=True)
        final_result['time'] = pd.to_datetime(final_result['time'])
        final_result['date'] = final_result['time'].dt.date
        final_result['date'] = pd.to_datetime(final_result['date'])
        #
        final_result = final_result.drop(columns=['地区', 'time'])
        final_result = final_result.loc[:, ~final_result.columns.str.startswith('Unnamed')]
        print('nc插值结果：', final_result)
        # =========================================================


        name_mapping = {
            '地区': 'area',
            '图像日期': 'time',
            '东经': 'lon',
            '北纬': 'lat'

        }
        fire.rename(columns=name_mapping, inplace=True)

        # 转换时间列并添加时间特征
        fire['time'] = pd.to_datetime(fire['time'])
        fire['year'] = fire['time'].dt.year
        fire['month'] = fire['time'].dt.month
        fire['day'] = fire['time'].dt.day
        # fire['hour'] = fire['time'].dt.hour
        fire['date'] = fire['time'].dt.date

        fire['date'] = pd.to_datetime(fire['date'])
        # print(fire['date'])
        # sta = pd.read_csv('dataset.csv')
        sta = self.merge
        print(sta)
        sta['date'] = pd.to_datetime(sta[['year', 'month', 'day']])
        merged_data = pd.merge(sta, fire, left_on='date', right_on='date', how='inner')
        print(merged_data)
        merged_data['distance'] = np.sqrt((merged_data['lon_x'] - merged_data['lon_y']) ** 2 +
                                          (merged_data['lat_x'] - merged_data['lat_y']) ** 2)

        merged_data = merged_data.sort_values(by=['date', 'area', 'distance'])
        merged_data = merged_data.drop_duplicates()

        column_names = ['Alti', 'TEM_Max', 'TEM_Min',
                        'RHU_Min', 'PRE_Time_2020', 'Snow_Depth', 'WIN_S_Max',
                        'yth1', 'yth10', 'yth100', 'yth1000', 'kb', 'erc', 'sc', 'bi', 'p',
                        'FFMC', 'DMC', 'DC', 'FWI', 'ISI', 'BUI', 'DSR', 'FFDI', 'ic']
        numeric_cols = merged_data[column_names].select_dtypes(include=[np.number]).columns
        merged_data = merged_data[(merged_data[numeric_cols] <= 9000).all(axis=1)]
        merged_data = merged_data.groupby(['date','lon_y','lat_y']).head(200).reset_index(drop=True)

        column_names_list = merged_data.columns.tolist()
        print(column_names_list)

        final_interpolated_results = None
        # print(merged_data)
        # 循环遍历每个特征并进行插值
        for column_name in column_names:

            grouped_data = list(merged_data.groupby(['date','lon_y','lat_y']))
            args_list = [(group, column_name) for _, group in grouped_data]
            with Pool(processes=20) as pool:
                interpolated_results_list = pool.starmap(interp.print_and_perform_kriging, args_list)
            interpolated_results = pd.concat(interpolated_results_list)

            # # 对当前特征进行插值
            # interpolated_results = merged_data.groupby('date','lon_y','lat_y').apply(
            #     lambda group: interp.perform_kriging_batchwise(group, column_name))
            # print(interpolated_results)
            # if interpolated_results.empty:
            #     print(f"跳过空的插值结果，列名为 {column_name}")
            #     continue

            # 重置索引
            interpolated_results.reset_index(drop=True, inplace=True)
            # 将插值结果合并到最终的DataFrame中
            if final_interpolated_results is None:
                final_interpolated_results = interpolated_results
            else:
                common_columns = ['date', 'lon', 'lat', 'year', 'month', 'day', 'area']
                final_interpolated_results = final_interpolated_results.merge(interpolated_results, on=common_columns)
        final_interpolated_results = pd.merge(final_interpolated_results, final_result, on=['date', 'lon', 'lat'])
        # 将最终插值结果保存到CSV文件中
        final_interpolated_results.to_csv(self.interp_result)
        # print(final_interpolated_results)

        return final_interpolated_results


# if __name__ == "__main__":
#     stime = str(20150408)
#     etime = str(20150709)
#     area = ['辽宁省', '黑龙江省', '吉林省']
#     datapprocess = Dataprocess(stime=stime, etime=etime,area =area)
#     obs_data = datapprocess.obs_data()
#     model_data = datapprocess.model_data()
#     merge_data = datapprocess.merge_data()
#     interp_data = datapprocess.interp()
                