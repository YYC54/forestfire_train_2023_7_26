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
config = configparser.ConfigParser()
print(sys.path[0])
config.read(os.path.join(sys.path[0], 'configs.ini'))
# config.read('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/configs.ini')



class Train(object):
    def __init__(self,pred_data,area):

        self.train_data = None #训练集
        self.pred_data = pred_data
        self.area = area
        self.train_png =config.get('TRAIN_PATH','result_png')
        self.model_path = config.get('TRAIN_PATH','model_path')
        self.cloumn = None
        if self.area == ['辽宁省', '黑龙江省', '吉林省']:
            self.diqu = 'Northeast'
            self.dataframe = config.get('TRAIN_PATH','train_dataset_north')
        else:
            self.diqu = 'Southwest'
            self.dataframe = config.get('TRAIN_PATH', 'train_dataset_west')
            self.dataframe1 = config.get('TRAIN_PATH','train_dataset_west1')
        self.result = config.get('TRAIN_PATH','result_path')
        self.train_chose = config.get('TRAIN_PATH','TRAIN')
    def _feature(self):
        """特征工程
        :return:
        self.train_data:训练数据集
        self.dataframe:预测数据集
        """
        data = pd.read_csv(self.dataframe)
        print(data.dtypes)
        data = data.drop(columns=['Unnamed: 0'])
        data = train_.clean(data) #清洗数据
        feats_name = train_.select_features(data, 0.1)
        # 对分类特征进行独热编码

        data = pd.get_dummies(data, columns=['area'])
        # print(data.dtypes)
        data = train_.mix_feature(data)
        # print(data.columns)
        data = data[(data['year'] != 2017) & (data['month'] != 5)&(data['month'] != 4)]
        # pred_data = data[(data['year']==2017)]
        self.train_data = data
    def _train(self):
        """训练
        :return:

        """
        train_data = self.train_data.dropna()

        # 定义特征和目标变量
        X = train_data.drop('fire', axis=1)

        y = train_data['fire']
        X['date'] = pd.to_datetime(X['date'])
        X["date"] = X["date"].apply(lambda x: x.timestamp())
        X, y = shuffle(X, y, random_state=0)

        strategy = {1: int(len(y[y == 0]) * 0.4)}
        smote = SMOTE(sampling_strategy=strategy)  # 建立SMOTE模型对象
        X, y = smote.fit_resample(X, y)  # 输入数据并作过采样处理

        print(y.describe())
        # 分割数据集为训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.cloumn = X_train.columns.tolist()

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        joblib.dump(scaler, f'{self.model_path}/{self.diqu}_scaler.pkl')

        param_grid = {
            'max_depth': [2 , 3, 4, 5], #max_depth: 控制树的最大深度。减小这个值可以使模型更简单，也就是剪枝
            'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
            'n_estimators': [50,80,100, 200, 300,400,500],
            'min_child_weight': [1, 2, 3, 4],#min_child_weight: 最小叶子节点样本权重和。这个参数越大，算法越保守。
            'gamma': [0, 0.1, 0.2],  # 更大的 gamma gamma（或者叫 min_split_loss）: 叶子节点进一步分区所需的最小损失减少。这个值越大，算法越保守
            'alpha': [0, 0.1, 0.2],  # 更大的 alpha alpha: L1 正则化项。增加这个值会使模型更简单。binary:logistic
            'lambda': [1, 2, 3, 4],  # 更大的 lambda L2 正则化项。增加这个值会使模型更简单。binary:hinge
            'objective': ['binary:hinge']
        }

        grid_search = GridSearchCV(estimator=xgb.XGBClassifier(random_state = 33),
                                   param_grid=param_grid,
                                   scoring='f1',#accuracy
                                   cv=5,
                                   n_jobs=-1,
                                   error_score='raise')
        # num_pos = np.sum(y_train == 1)
        # num_neg = np.sum(y_train == 0)
        # weight_for_pos = num_neg / len(y_train)
        # weight_for_neg = num_pos / len(y_train)
        # weights = np.where(y_train == 1, weight_for_pos, weight_for_neg)  # 平衡正负样本权重, sample_weight=weights

        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        print(f"Best Parameters: {best_params}")

        # 使用最佳参数训练模型
        model = xgb.XGBClassifier(**best_params)
        # 计算权重
        # num_pos = np.sum(y_train == 1)
        # num_neg = np.sum(y_train == 0)
        # weight_for_pos = num_neg / len(y_train)
        # weight_for_neg = num_pos / len(y_train)
        # weights = np.where(y_train == 1, weight_for_pos, weight_for_neg)#平衡正负样本权重,
        #                   sample_weight=weights

        # 训练模型并使用早停
        evals = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_metric="auc", eval_set=evals, early_stopping_rounds=50)

        # model.fit(X_train, y_train,sample_weight=weights)

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

        if self.diqu =='Northeast':
            plt.savefig(f'{self.train_png}/{self.diqu}_ROC.png')
            model_path = f'{self.model_path}/{self.diqu}_fire_risk_model.xgb'
        else:
            plt.savefig(f'{self.train_png}/{self.diqu}_ROC.png')
            model_path = f'{self.model_path}/{self.diqu}_fire_risk_model.xgb'

        # 保存模型
        model.save_model(model_path)

    def _pred(self):
        final_pred_data = self.pred_data.copy()
        # self.pred_data = self.pred_data.drop(columns=['Unnamed: 0'])
        print(self.pred_data)
        # # 创建DMatrix对象
        # dpred = xgb.DMatrix(X_pred)
        model = xgb.Booster()
        model.load_model(f'{self.model_path}/{self.diqu}_fire_risk_model.xgb')

        # self.pred_data['year'] = self.pred_data['year'].astype('int64')
        # self.pred_data['month'] = self.pred_data['year'].astype('int64')
        # self.pred_data['day'] = self.pred_data['year'].astype('int64')
        #特征工程
        self.pred_data = pd.get_dummies(self.pred_data, columns=['area'])

        # one_hot = OneHotEncoder(sparse=False,handle_unknown='ignore' )
        # self.pred_data = one_hot.fit_transform(self.pred_data)
        for area_name in self.area:
            column_name = f'area_{area_name}'
            if column_name not in self.pred_data.columns:
                self.pred_data[column_name] = 0  # 如果列不存在，创建列并将所有值设置为0
        # print('>>>>>>>>>>>>>>>>>',self.pred_data.columns)
        self.pred_data = pred_func.mix_feature(self.pred_data)
        print('>>>>>>>>>>>>>>>>>', self.pred_data.columns)
        # self.pred_data = pred_func.stander(self.pred_data,f'{self.model_path}/{self.diqu}_scaler.pkl')


        # 更新Snow_Depth列
        try:
            self.pred_data.loc[self.pred_data['Snow_Depth'] < 0, 'Snow_Depth'] = 0
        except:
            self.pred_data['Snow_Depth'] = 0
        # 更新PRE_Time_2020列
        self.pred_data.loc[self.pred_data['PRE_Time_2020'] < 0, 'PRE_Time_2020'] = 0
        self.pred_data.loc[self.pred_data['FFDI'] < 0, 'FFDI'] = 0

        self.pred_data = self.pred_data[self.cloumn]
        self.pred_data = pred_func.stander(self.pred_data, f'{self.model_path}/{self.diqu}_scaler.pkl')
        # self.pred_data = pred_func.mix_feature(self.pred_data)

        X_pred = self.pred_data
        X_pred = xgb.DMatrix(X_pred)
        # 获取当前时间
        now = datetime.now()
        # 提取年、月和日
        year = now.year
        month = now.month
        day = now.day

        # 使用保存的模型进行预测
        predictions = model.predict(X_pred)
        final_pred_data['prediction'] = predictions

        if self.diqu == 'Northeast':
            final_pred_data.to_csv(f'{self.result}/Northeast/{year}{month}{day}.csv')
        else:
            final_pred_data.to_csv(f'{self.result}/Southwest/{year}{month}{day}.csv')

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

