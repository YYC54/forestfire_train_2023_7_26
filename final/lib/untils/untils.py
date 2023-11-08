import glob
import pandas as pd
# import tensorflow as tf
import os
from typing import Generator
import re
import numpy as np

def find_txt_files(directory, date):
    txt_files = []
    for root, dirs, files in os.walk(directory):
        # print("Root:", root)
        # print("Dirs:", dirs)
        # print("Files:", files)
        for file in files:
            # print(file)
            if file.endswith('.txt') and str(date) == file.split('.')[0]:
                txt_path = os.path.join(root, file)
                # print(txt_path)
                txt_files.append(txt_path)
                # print("Found:", txt_path)
    # print("All txt files:", txt_files)
    print(txt_files)
    return txt_files


def read_txt_file(file_path):
    # 从文件名中提取年、月、日
    # date_str = os.path.basename(file_path).split('.')[0]  # 例如，得到"20100101"
    # year = int(date_str[:4])
    # month = int(date_str[4:6])
    # day = int(date_str[6:])

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # # 将年、月、日添加为新列
    # df['year'] = year
    # df['month'] = month
    # df['day'] = day

    return df

def txt_file_generator(file_list: list) -> Generator[pd.DataFrame, None, None]:
    for txt_file in file_list:
        yield read_txt_file1(txt_file)


def read_txt_file1(file_path):
    # 从文件名中提取年、月、日
    date_str = os.path.basename(file_path).split('.')[0]  # 例如，得到"20100101"
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:])

    df = pd.read_csv(file_path, delim_whitespace=True, header=None)

    # 将年、月、日添加为新列
    df['year'] = year
    df['month'] = month
    df['day'] = day

    return df