import pandas as pd
import numpy as np


dataframe = pd.read_csv('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/train_dataset/interpolated_results_west.csv')
dataframe1 = pd.read_csv('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/train_dataset/west.csv')
# dataframe1.drop(columns=['地区','Unnamed: 0','fire'])
name_mapping = {

            '图像日期': 'time',
            '东经': 'lon',
            '北纬': 'lat'

        }
dataframe1.rename(columns=name_mapping, inplace=True)
dataframe1['time'] = pd.to_datetime(dataframe1['time'])
dataframe1['date'] = dataframe1['time'].dt.date
dataframe1 = dataframe1.drop(columns=['地区','Unnamed: 0','fire','time'])
meergeddata = pd.merge(dataframe,dataframe1,on=['date','lon','lat'])
print(meergeddata.head())
