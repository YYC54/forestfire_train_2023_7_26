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


def mix_feature(data):
    # Selecting numerical features for interaction term creation
    # numerical_features = data.select_dtypes(include=['float64', 'int64', 'datetime64','object']).drop(columns=['date', 'year', 'month', 'day', 'fire', 'lon', 'lat'])
    numerical_features = data.drop(columns=['date', 'year', 'month', 'day', 'fire', 'lon', 'lat'])
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
    #data.loc[data['Snow_Depth'] < 0, 'Snow_Depth'] = 0
    # 更新PRE_Time_2020列
    data.loc[data['PRE_Time_2020'] < 0, 'PRE_Time_2020'] = 0
    data.loc[data['FFDI'] < 0, 'FFDI'] = 0
    data = data.dropna()
    return data

def fetures(data):
    data = pd.read_csv(data)
    print(data.dtypes)
    data = data.drop(columns=['Unnamed: 0','area'])
    data = clean(data)  # 清洗数据

    #feats_name = select_features(data, 0.1)
    # 对分类特征进行独热编码

    #data = pd.get_dummies(data, columns=['area'])
    # print(data.dtypes)
    data.drop(['可燃物含水率', '土壤湿度', 'Snow_Depth', 'Alti'], axis=1, inplace=True)
    data = mix_feature(data)
    # print(data.columns)
    #data = data[(data['year'] != 2017) & (data['month'] != 5) & (data['month'] != 4)]
    # pred_data = data[(data['year']==2017)]
    return data



def train_code(data,model_path,train_png,diqu):
    train_data = data.dropna()
    # 定义特征和目标变量
    X = train_data.drop('fire', axis=1)
    y = train_data['fire']
    X['date'] = pd.to_datetime(X['date'])
    X["date"] = X["date"].apply(lambda x: x.timestamp())
    X, y = shuffle(X, y, random_state=0)

    strategy = {1: int(len(y[y == 0]) * 0.3)}
    smote = SMOTE(sampling_strategy=strategy)  # 建立SMOTE模型对象
    X, y = smote.fit_resample(X, y)  # 输入数据并作过采样处理

    print(y.describe())
    # 分割数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cloumn = X_train.columns.tolist()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    joblib.dump(scaler, f'{model_path}/{diqu}_scaler.pkl')

    param_grid = {
        'max_depth': [2, 3, 4, 5],# max_depth: 控制树的最大深度。减小这个值可以使模型更简单，也就是剪枝'n_estimators': [50,80,100, 200, 300,400,500]
        'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
        'n_estimators': [50, 80, 100, 200, 300, 400, 500],
        'min_child_weight': [1, 2, 3, 4],  # min_child_weight: 最小叶子节点样本权重和。这个参数越大，算法越保守。
        'gamma': [0, 0.1, 0.2],  # 更大的 gamma gamma（或者叫 min_split_loss）: 叶子节点进一步分区所需的最小损失减少。这个值越大，算法越保守
        'alpha': [0, 0.1, 0.2],  # 更大的 alpha alpha: L1 正则化项。增加这个值会使模型更简单。binary:logistic
        'lambda': [1, 2, 3, 4],  # 更大的 lambda L2 正则化项。增加这个值会使模型更简单。binary:hinge
        'objective': ['binary:hinge']
        # 'max_depth': [5],# max_depth: 控制树的最大深度。减小这个值可以使模型更简单，也就是剪枝'n_estimators': [50,80,100, 200, 300,400,500]
        # 'learning_rate': [0.1],
        # 'n_estimators': [500],
        # 'min_child_weight': [3],  # min_child_weight: 最小叶子节点样本权重和。这个参数越大，算法越保守。
        # 'gamma': [0.2],  # 更大的 gamma gamma（或者叫 min_split_loss）: 叶子节点进一步分区所需的最小损失减少。这个值越大，算法越保守
        # 'alpha': [0],  # 更大的 alpha alpha: L1 正则化项。增加这个值会使模型更简单。binary:logistic
        # 'lambda': [3],  # 更大的 lambda L2 正则化项。增加这个值会使模型更简单。binary:hinge
        # 'objective': ['binary:hinge']

    }

    grid_search = GridSearchCV(estimator=xgb.XGBClassifier(random_state=33),
                               param_grid=param_grid,
                               scoring='f1',  # accuracy
                               cv=5,
                               n_jobs=-1,
                               error_score='raise')
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    weight_for_pos = num_neg / len(y_train)
    weight_for_neg = num_pos / len(y_train)
    # weights = np.where(y_train == 1, weight_for_pos, weight_for_neg)  # 平衡正负样本权重, sample_weight=weights
    weights = np.where(y_train == 1, 2, 1)
    grid_search.fit(X_train, y_train, sample_weight=weights)

    best_params = grid_search.best_params_
    print(f"Best Parameters: {best_params}")

    # 使用最佳参数训练模型
    model = xgb.XGBClassifier(**best_params)
    # 计算权重
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    weight_for_pos = num_neg / len(y_train)
    weight_for_neg = num_pos / len(y_train)
    weights = np.where(y_train == 1, 2, 1)
    # weights = np.where(y_train == 1, weight_for_pos, weight_for_neg)#平衡正负样本权重,
    #                   sample_weight=weights

    # # 训练模型并使用早停
    evals = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_metric="auc", eval_set=evals, early_stopping_rounds=50, sample_weight=weights)

    # 收集训练过程中的评估结果
    evals_result = model.evals_result()

    # 绘制曲线图
    train_auc = evals_result['validation_0']['auc']
    test_auc = evals_result['validation_1']['auc']
    epochs = len(train_auc)
    x_axis = range(0, epochs)

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_auc, label='Train')
    plt.plot(x_axis, test_auc, label='Test')
    plt.legend()
    plt.ylabel('AUC')
    plt.xlabel('Epochs')
    plt.title('XGBoost AUC')
    plt.savefig(f'{train_png}/auc.png')
    #plt.show()

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)  # 进行训练集模型预测

    print("-----------------训练集性能评估-------------------")
    # 进行模型评估：通过准确率、查准率、查全率、F1值，并分为训练集评估和验证集评估
    print("训练集准确率: {:.2f}%".format(accuracy_score(y_train, y_train_pred) * 100))
    print("训练集查准率: {:.2f}%".format(precision_score(y_train, y_train_pred) * 100))  # 打印训练集查准率
    print("训练集查全率: {:.2f}%".format(recall_score(y_train, y_train_pred) * 100))  # 打印训练集查全率
    print("训练集F1值: {:.2f}%".format(f1_score(y_train, y_train_pred) * 100))  # 打印训训练集F1值
    print("----------------验证集性能评估--------------------")
    print("验证集准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
    print("验证集查准率: {:.2f}%".format(precision_score(y_test, y_pred) * 100))  # 打印验证集查准率
    print("验证集查全率: {:.2f}%".format(recall_score(y_test, y_pred) * 100))  # 打印验证集查全率
    print("验证集F1值: {:.2f}%".format(f1_score(y_test, y_pred) * 100))  # 打印验证集F1值

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # plt.rcParams['font.sans-serif'] = ['Arial']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # plt.savefig(f'{self.train_png}/ROC.png')


    plt.savefig(f'{train_png}/{diqu}_ROC.png')

    model_pp = f'{model_path}/{diqu}_fire_risk_model.xgb'
    model.save_model(model_pp)
    return cloumn