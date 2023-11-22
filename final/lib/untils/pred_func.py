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
import joblib

def mix_feature(data):
    # Selecting numerical features for interaction term creation
    numerical_features = data.select_dtypes(include=['float64', 'int64' ,'datetime64','object']).drop(columns=['date', 'year', 'month', 'day','lon', 'lat'])

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
    data.loc[data['Snow_Depth'] < 0, 'Snow_Depth'] = 0
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
