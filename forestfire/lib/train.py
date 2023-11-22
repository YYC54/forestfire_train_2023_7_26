import sys
sys.path.append(".")
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import meteva as mem
from meteva.method import pc
from sklearn.feature_selection import mutual_info_classif
import numpy as np
import joblib
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
from lib.untils import train_,pred_func
import configparser
from datetime import datetime
import os
from imblearn.over_sampling import SMOTE
import xarray as xr
config = configparser.ConfigParser()
print(sys.path[0])
config.read(os.path.join(sys.path[0], 'configs.ini'))
# config.read('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/configs.ini')



class Train(object):
    def __init__(self,merge_ne,merge_sw,stime):
        self.stime = stime
        self.train_data_ne = None #训练集
        self.train_data_sw = None
        self.merge_ne = merge_ne
        self.merge_sw = merge_sw

        self.train_png =config.get('TRAIN_PATH','result_png')
        self.model_path = config.get('TRAIN_PATH','model_path')
        self.cloumn = None

        self.dataframe_ne = config.get('TRAIN_PATH','train_dataset_north')
        self.dataframe_sw = config.get('TRAIN_PATH', 'train_dataset_west')

        self.result = config.get('TRAIN_PATH','result_path')
        self.train_chose = config.get('TRAIN_PATH','TRAIN')
    def _feature(self):
        """特征工程
        :return:
        self.train_data:训练数据集
        self.dataframe:预测数据集
        """
        self.train_data_ne = train_.fetures(self.dataframe_ne)

        self.train_data_sw = train_.fetures(self.dataframe_sw)

    def _train(self):
        """训练
        :return:

        """
        self.cloumn = train_.train_code(self.train_data_ne,self.model_path,self.train_png,'Northeast')
        self.cloumn = train_.train_code(self.train_data_sw, self.model_path, self.train_png, 'Southwest')

        column_list = self.cloumn
        # 将列表数据写入文件
        with open(f'{self.result}/column_list.txt', 'w') as file:
            for item in column_list:
                file.write("%s\n" % item)

    def _pred(self):

        final_pred_data_ne = self.merge_ne.copy()
        final_pred_data_sw = self.merge_sw.copy()
        # 读取文件数据到列表
        with open(f'{self.result}/column_list.txt', 'r') as file:
            column_list = file.readlines()

        # 去除每行末尾的换行符
        column = [item.strip() for item in column_list]

        self.merge_ne = pred_func.feature(self.merge_ne,column,self.model_path,'Northeast')
        self.merge_sw = pred_func.feature(self.merge_sw, column, self.model_path, 'Southwest')
        predictions_ne = pred_func.pred(self.merge_ne,self.model_path,'Northeast')
        predictions_sw = pred_func.pred(self.merge_sw, self.model_path, 'Southwest')
        final_pred_data_ne['fire'] = predictions_ne
        final_pred_data_sw['fire'] = predictions_sw

        # 提取年份（前四个字符）
        year = self.stime[:4]
        # 提取月份（第五个和第六个字符）
        month = self.stime[4:6]
        # 提取日（最后两个字符）
        day = self.stime[6:]
        directory_ne = f'{self.result}/Northeast/{year}/{month}/{day}'
        if not os.path.exists(directory_ne):
            os.makedirs(directory_ne)
        final_pred_data_ne.to_csv(f'{directory_ne}/{self.stime}.csv')

        directory_sw = f'{self.result}/Southwest/{year}/{month}/{day}'
        if not os.path.exists(directory_sw):
            os.makedirs(directory_sw)
        final_pred_data_sw.to_csv(f'{directory_sw}/{self.stime}.csv')

        grouped_data_ne = final_pred_data_ne.groupby('time')
        for time, df in grouped_data_ne:
            # 构建文件名
            # data = transfer(df)
            df =df[['lat','lon','fire']]
            filename = f'{self.result}/Northeast/{year}/{month}/{day}/{time}.nc'  # 使用时间作为文件名
            # 调用函数来保存为 NetCDF 文件
            pred_func.dataframe_to_netcdf(df, filename)


        grouped_data_sw = final_pred_data_sw.groupby('time')
        for time, df in grouped_data_sw:
            # 构建文件名
            # data = transfer(df)
            df = df[['lat', 'lon', 'fire']]
            filename = f'{self.result}/Southwest/{year}/{month}/{day}/{time}.nc'  # 使用时间作为文件名
            # 调用函数来保存为 NetCDF 文件
            pred_func.dataframe_to_netcdf(df, filename)

# if __name__ == "__main__":
#
#     pred_data = pd.read_csv('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/interpolated_results.csv')
#     area = ['辽宁省', '黑龙江省', '吉林省']
#
#     train = Train(pred_data=pred_data,area=area)
#
#     train._feature()
#     # train._train()
#     train._pred()

