import sys
sys.path.append(".")
import glob
import pandas as pd
import os
from typing import Generator
import re
from lib.untils import untils
from datetime import datetime
from lib.dataprocess import Dataprocess
from lib.train import Train
import configparser
config = configparser.ConfigParser()
print(sys.path[0])
config.read(os.path.join(sys.path[0], 'configs.ini'))
# config.read('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/configs.ini')

class Forest(object):
    def __init__(self, stime , bound):

        self.stime = stime
        self.bound = bound
        self.merge_sw = None
        self.merge_ne = None
        self.train_chose = config.get('TRAIN_PATH', 'TRAIN')

    def dataset(self):
        dataprocess = Dataprocess(stime=self.stime,  bound=self.bound)
        obs_data = dataprocess.obs_data()  # 气象数据处理
        model_data = dataprocess.model_data()  # 模型数据处理
        self.merge_ne,self.merge_sw = dataprocess.merge_data()  # 实况 模型数据匹配
        # self.pred_data = dataprocess.interp()  # 匹配好的数据插值到火点

    def train_pred(self):

        train = Train(self.merge_ne,self.merge_sw,self.stime)

        if self.train_chose =='1':

            train._feature()
            train._train()
            train._pred()
        else:
            # train._feature()
            # train._train()
            train._pred()


if __name__ == "__main__":
    time = '20231113'
    # time = str(sys.argv[1])
    bound = 10
    forest = Forest(stime=time,bound = bound)
    forest.dataset()
    forest.train_pred()