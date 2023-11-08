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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr


def mix_feature(data):
    # Selecting numerical features for interaction term creation
    numerical_features = data.select_dtypes(include=['float64', 'int64', 'datetime64','object']).drop(columns=['date', 'year', 'month', 'day', 'fire', 'lon', 'lat'])

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

def select_features(df,thus):
    selected_features = []
    print(df.describe())
    # Loop through each feature column to calculate the Point-Biserial Correlation Coefficient
    for column in df.columns:
        if column not in ['fire', 'date', 'year', 'month', 'day', 'lon', 'lat','area']:
            coeff, _ = pointbiserialr(df[column], df['fire'])
            if abs(coeff) >= thus:
                selected_features.append(column)
    
    print(f'Features with Point-Biserial Correlation Coefficient > {thus}: {selected_features}')
    
    # Combine selected features with other necessary columns
    feats_name = ['fire', 'date', 'year', 'month', 'day', 'lon', 'lat','area'] + selected_features
    
    return feats_name

def process_outlier(df,feats):
    _ = df[feats[0:]]
    # 1. 计算每个气象要素的平均值和标准差
    mean_values = _.mean()
    std_values = _.std()
    # 2. 根据三倍标准差法，设定异常值的阈值
    threshold = 3 * std_values
    # 3. 遍历每个气象要素的数值，将超过设定阈值的值标记为异常值
    is_outlier = (_ > mean_values + threshold) | (_ < mean_values - threshold)
    # 4. 对于标记为异常值的数据，可以根据需要选择删除、替换或进行插补处理
    # 假设你选择删除异常值
    df = df[~is_outlier.any(axis=1)]
#     df = df.query('sw_max1<20 and sw_max2<20')
     
    return df
def clean(data):
    '''

    :param data:
    :return: clean data
    '''
    data.loc[data['Snow_Depth'] < 0, 'Snow_Depth'] = 0
    # 更新PRE_Time_2020列
    data.loc[data['PRE_Time_2020'] < 0, 'PRE_Time_2020'] = 0
    data.loc[data['FFDI'] < 0, 'FFDI'] = 0
    data = data.dropna()
    return data
