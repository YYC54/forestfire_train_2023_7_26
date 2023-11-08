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
    def __init__(self, stime, etime, area):

        self.stime = datetime.strptime(stime, '%Y%m%d')
        self.etime = datetime.strptime(etime, '%Y%m%d')
        self.area = area
        self.pred_data = None
        self.train_chose = config.get('TRAIN_PATH', 'TRAIN')
    def dataset(self):
        dataprocess = Dataprocess(stime=self.stime, etime=self.etime, area=self.area)
        obs_data = dataprocess.obs_data()  # 实况数据处理
        model_data = dataprocess.model_data()  # 模型数据处理
        merge_data = dataprocess.merge_data()  # 实况 模型数据匹配
        self.pred_data = dataprocess.interp()  # 匹配好的数据插值到火点

    def train_pred(self):
        print(self.pred_data)
        train = Train(pred_data=self.pred_data, area=self.area)
        if self.train_chose =='1':
            train._feature()
            train._train()
            train._pred()
        else:
            train._feature()
            # train._train()
            train._pred()


if __name__ == "__main__":
    stime = str(20170411)
    etime = str(20170514)
    # area = ['辽宁省', '黑龙江省', '吉林省']
    area =['云南省','贵州省','四川省','重庆市']

    forest = Forest(stime=stime, etime=etime, area=area)
    forest.dataset()
    forest.train_pred()