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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr
import joblib
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import configparser
from datetime import datetime
import os
from imblearn.over_sampling import SMOTE
import xarray as xr

def mix_feature(data):
    # Selecting numerical features for interaction term creation
    # numerical_features = data.select_dtypes(include=['float64', 'int64' ,'datetime64','object']).drop(columns=['date', 'year', 'month', 'day','lon', 'lat'])
    numerical_features = data.drop(
        columns=['date', 'year', 'month', 'day', 'lon', 'lat'])

    # Creating interaction terms using PolynomialFeatures
    polynomial_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = polynomial_features.fit_transform(numerical_features)

    # Creating a DataFrame to hold the new interaction features
    interaction_features_df = pd.DataFrame(interaction_features)

    # Rename interaction features to ensure uniqueness
    new_feature_names = [f'feature_{i}' for i in range(interaction_features_df.shape[1])]
    interaction_features_df.columns = new_feature_names

    # Merging the interaction features with the original dataset
    data_with_interactions = pd.concat([data, interaction_features_df], axis=1)
    return data_with_interactions


def select_features(df, thus):
    selected_features = []
    print(df)
    # Loop through each feature column to calculate the Point-Biserial Correlation Coefficient
    for column in df.columns:
        if column not in ['fire', 'date', 'year', 'month', 'day', 'lon', 'lat']:
            coeff, _ = pointbiserialr(df[column], df['fire'])
            if abs(coeff) >= thus:
                selected_features.append(column)

    print(f'Features with Point-Biserial Correlation Coefficient > {thus}: {selected_features}')

    # Combine selected features with other necessary columns
    feats_name = ['date', 'year', 'month', 'day', 'lon', 'lat'] + selected_features

    return feats_name
def clean(data):
    '''

    :param data:
    :return: clean data
    '''
    # 更新PRE_Time_2020列
    data.loc[data['PRE_Time_2020'] < 0, 'PRE_Time_2020'] = 0
    data.loc[data['FFDI'] < 0, 'FFDI'] = 0
    return data

def stander(data,sacler):

    data['date'] = pd.to_datetime(data['date'])
    data["date"] = data["date"].apply(lambda x: x.timestamp())
    scaler = joblib.load(sacler)
    features_scaled = scaler.transform(data)
    return features_scaled

def feature(data,cloumn,model_path,diqu):
    # 特征工程
    data['time'] = pd.to_datetime(data['time'])
    data['year'] = data['time'].dt.year
    data['month'] = data['time'].dt.month
    data['day'] = data['time'].dt.day
    data['date'] = data['time'].dt.date
    # 删除原始的 'time' 列
    data.drop('time', axis=1, inplace=True)
    #data = pd.get_dummies(data, columns=['area'])

    # one_hot = OneHotEncoder(sparse=False,handle_unknown='ignore' )
    # self.pred_data = one_hot.fit_transform(self.pred_data)
    #for area_name in self.area:
        #column_name = f'area_{area_name}'
        #if column_name not in self.pred_data.columns:
            #self.pred_data[column_name] = 0  # 如果列不存在，创建列并将所有值设置为0
    # print('>>>>>>>>>>>>>>>>>',self.pred_data.columns)
    data = mix_feature(data)
    print('>>>>>>>>>>>>>>>>>', data.columns)
    # self.pred_data = pred_func.stander(self.pred_data,f'{self.model_path}/{self.diqu}_scaler.pkl')

    # 更新Snow_Depth列

    # 更新PRE_Time_2020列
    data.loc[data['PRE_Time_2020'] < 0, 'PRE_Time_2020'] = 0
    data.loc[data['FFDI'] < 0, 'FFDI'] = 0

    data = data[cloumn]
    data = stander(data, f'{model_path}/{diqu}_scaler.pkl')
    return data

def pred(data,model_path,diqu):
    model = xgb.Booster()
    model.load_model(f'{model_path}/{diqu}_fire_risk_model.xgb')
    X_pred = data
    X_pred = xgb.DMatrix(X_pred)

    # 使用保存的模型进行预测
    predictions = model.predict(X_pred)
    return predictions

def transfer(data):
    element = 'fire'
    grid_data_ne = data.pivot(index='lat', columns='lon', values=element)

    return grid_data_ne


def dataframe_to_netcdf(df, filename):
    # 确保 'lat' 和 'lon' 是 DataFrame 的索引
    if not {'lat', 'lon'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'lat' and 'lon' columns")

    # df.set_index(['lat', 'lon'], inplace=True)
    grid = transfer(df)
    # 将 DataFrame 转换为 xarray DataArray
    data_array = xr.DataArray(grid, dims=('lat', 'lon'))

    # 创建 xarray Dataset
    ds = xr.Dataset({'fire': data_array})

    # 保存为 NetCDF 文件
    ds.to_netcdf(filename)
