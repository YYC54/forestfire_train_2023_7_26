import pandas as pd
from datetime import timedelta,datetime
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import warnings
from pandas.errors import DtypeWarning
from scipy.spatial import cKDTree
import numpy as np
from pykrige.ok import OrdinaryKriging
from numpy.linalg import LinAlgError
import os
import xarray as xr
import csv
warnings.simplefilter(action='ignore', category=DtypeWarning)
warnings.filterwarnings('ignore')



def print_and_perform_kriging(group, column_name):
    # group, column_name = args
    print("Group data:")# 打印当前 group 的内容
    print(group)
    return perform_kriging_batchwise(group, column_name)  # 执行插值


def perform_kriging_batchwise(group, column_name):
    try:
        group = group.drop_duplicates(subset=['lon_x', 'lat_y'])

        # 重新索引
        group.reset_index(drop=True, inplace=True)

        if column_name == 'ic':
            group[column_name] = pd.to_numeric(group[column_name], errors='coerce')

        # 从数据组中提取经度、纬度和目标特征值
        lon = group['lon_x'].values
        lat = group['lat_x'].values
        target_values = group[column_name].values
        # print(len(lon), len(lat), len(target_values))

        # min_points_required = 3
        # if len(lon) < min_points_required or len(lat) < min_points_required or len(target_values) < min_points_required:
        #     print(f"跳过日期为 {group['date'].iloc[0]} 和列为 {column_name} 的数据组（点数太少）")
        #     return pd.DataFrame()  # 返回一个空的 DataFrame

        # 检查 NaN 值
        if np.isnan(lon).any() or np.isnan(lat).any() or np.isnan(target_values).any():
            print(f"跳过日期为 {group['date'].iloc[0]} 和列为 {column_name} 的数据组（存在 NaN 值）")
            return pd.DataFrame()  # 返回一个空的 DataFrame
        # lon_y = group['lon_y'].values
        # lat_y = group['lat_y'].values
        lon_y = group['lon_y'].drop_duplicates().values
        # print(lon_y)
        lat_y = group['lat_y'].drop_duplicates().values

        # ok = OrdinaryKriging(
        #     lon, lat, target_values, variogram_model='linear',
        #     verbose=False, enable_plotting=False
        # )
        # z, ss = ok.execute('points', lon_y, lat_y)
        try:  # 尝试执行 OrdinaryKriging 和插值，可能会出现 "singular matrix" 问题
            ok = OrdinaryKriging(
                lon, lat, target_values, variogram_model='linear',
                verbose=False, enable_plotting=False
            )
            z, ss = ok.execute('points', lon_y, lat_y)
            # print(z)
        except LinAlgError:  # 捕获 "singular matrix" 异常
            print(f"Singular matrix error occurred for date group {group['date'].iloc[0]} and column {column_name}")
            return pd.DataFrame()  # 返回一个空的 DataFrame

        # 创建一个DataFrame用于保存插值结果
        interpolated_df = pd.DataFrame({
            'date': group['date'].iloc[0],  # 假设整个组的日期是相同的
            'lon': lon_y.ravel(),
            'lat': lat_y.ravel(),
            'year': group['year_y'].iloc[0],
            'month': group['month_y'].iloc[0],
            'day': group['day_y'].iloc[0],
            'area': group['area'].iloc[0],
            column_name: z.ravel()
        })

    except ValueError as e:
        print(f"An error occurred for date group {group['date'].iloc[0]} and column {column_name}: {str(e)}")
        return pd.DataFrame()  # or some other appropriate default value
    return interpolated_df

def interpolate_and_save(interp_data_path, data_path, new_column_name, output_file_path):
    # 读取火点数据
    with open(interp_data_path, mode='r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)
    # 获取列名作为header
    # header = fire.columns.tolist()
    # # 获取数据作为data
    # data = fire.values
    header.append(new_column_name)

    # 读入可燃物数据
    filtered_data = []

    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.nc'):
                year = int(file[:4])
                week = int(file[file.find('w')+1:file.find('.nc')])
                start_date = datetime(year, 1, 1) + timedelta(weeks=week - 1)
                # print("开始时间", start_date)
                end_date = datetime(year, 1, 1) + timedelta(weeks=week)
                if end_date.year != year:
                    end_date = datetime(year, 12, 31)  # 设置为年的最后一天
                # print("结束时间", end_date)

                file_path = os.path.join(subdir, file)
                # print("正在处理文件", file_path)
                ds = xr.open_dataset(file_path)

                for row in data:
                    image_date = row[2]  # 日期在第三列
                    date = datetime.strptime(str(image_date), '%Y-%m-%d %H:%M:%S')
                    if start_date <= date < end_date:
                        # print("处理日期", date)
                        lon1 = float(row[3])  # 经度在第四列
                        lat1 = float(row[4])  # 纬度在第五列
                        # 选择最接近的经度和纬度
                        ds_subset = ds.sel(lat=lat1, lon=lon1, method='nearest')
                        # 找到最接近点的索引
                        nearest_lat = ds_subset.lat.values
                        nearest_lon = ds_subset.lon.values
                        # 获取最接近点的值
                        nearest_value = ds.Band1.sel(lat=nearest_lat, lon=nearest_lon, method='nearest').values
                        # 将最接近值添加到原始数据的新列中
                        row.append(nearest_value)
                        # 将原始行添加到筛选数据中
                        filtered_data.append(row)



    # 将筛选后的数据保存为新的CSV文件
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # 写入标题行
        writer.writerows(filtered_data)

    # result = pd.DataFrame(filtered_data_all[:5],columns=header)
    #
    # if result.shape[1] > 5:
    #     result = result.drop(result.columns[5], axis=1)

    # return result
