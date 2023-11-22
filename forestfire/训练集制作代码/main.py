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
warnings.simplefilter(action='ignore', category=DtypeWarning)
warnings.filterwarnings('ignore')

def interpolate_and_save(fire, data_path, new_column_name):
    # 读取火点数据
    # with open(interp_data_path, mode='r') as file:
    #     reader = csv.reader(file)
    #     header = next(reader)
    #     data = list(reader)
    # 获取列名作为header
    header = fire.columns.tolist()
    # 获取数据作为data
    data = fire.values.tolist()
    header.append(new_column_name)

    # 读入可燃物数据
    filtered_data = []

    for subdir, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.nc'):
                year = int(file[:4])
                week = int(file[file.find('w')+1:file.find('.nc')])
                start_date = datetime(year, 1, 1) + timedelta(weeks=week - 1)
                print("开始时间", start_date)
                end_date = datetime(year, 1, 1) + timedelta(weeks=week)
                if end_date.year != year:
                    end_date = datetime(year, 12, 31)  # 设置为年的最后一天
                print("结束时间", end_date)

                file_path = os.path.join(subdir, file)
                print("正在处理文件", file_path)
                ds = xr.open_dataset(file_path)

                for row in data:
                    image_date = row[1]  # 日期在第三列
                    date = datetime.strptime(str(image_date), '%Y-%m-%d %H:%M:%S')
                    if start_date <= date <= end_date:
                        print("处理日期", date)
                        lon1 = float(row[2])  # 经度在第四列
                        lat1 = float(row[3])  # 纬度在第五列
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
    # with open(output_file_path, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(header)  # 写入标题行
    #     writer.writerows(filtered_data)
    result = pd.DataFrame(filtered_data, columns=header)

    return result


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
        print(len(lon), len(lat), len(target_values))

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
        print(lon_y)
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
            'fire': group['fire'].iloc[0],
            column_name: z.ravel()
        })

    except ValueError as e:
        print(f"An error occurred for date group {group['date'].iloc[0]} and column {column_name}: {str(e)}")
        return pd.DataFrame()  # or some other appropriate default value
    return interpolated_df

if __name__ == "__main__":
    fire = pd.read_excel('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/fire2010-2017.xls')
    fire['fire'] = 1
    # 时间序列扩充
    new_rows_list = []

    # 遍历所有唯一的火灾日期
    for fire_date in fire['图像日期'].unique():
        fire_rows = fire[fire['图像日期'] == fire_date]

        # 检查前两天和后一天的窗口
        for offset in range(-10,10):
            new_fire_date = pd.Timestamp(fire_date) + timedelta(days=offset)

            # 检查新日期是否已存在数据
            existing_rows = fire[fire['图像日期'] == new_fire_date]

            if existing_rows.empty:  # 如果不存在数据，则创建新行
                new_fire_rows = fire_rows.copy()
                new_fire_rows['图像日期'] = new_fire_date
                new_fire_rows['fire'] = 0
                new_rows_list.append(new_fire_rows)
            else:  # 如果存在数据，则使用原有数据
                new_rows_list.append(existing_rows)

    # 使用pandas.concat合并所有新行
    new_rows = pd.concat(new_rows_list, ignore_index=True)

    extended_fire_data = pd.concat([fire, new_rows], ignore_index=True).drop_duplicates()
    # extended_fire_data.to_csv('fire2010-2017_new.csv')

    provinces = ['云南省', '贵州省', '四川省','重庆市']
    # provinces = ['辽宁省', '黑龙江省', '吉林省']
    fire = extended_fire_data[extended_fire_data['地区'].isin(provinces)].copy()

    #----------------------------可燃物，土壤湿度插值------------------------------
    data_path = '/public/home/duyl_hx/firedata/a卫星反演/a可燃物含水率'
    new_column_name = "可燃物含水率"  # 新增列名
    result_water = interpolate_and_save(fire, data_path, new_column_name)
    data_path = '/public/home/duyl_hx/firedata/a卫星反演/c土壤湿度'
    new_column_name = "土壤湿度"  # 新增列名
    result_rh = interpolate_and_save(fire, data_path, new_column_name)
    final_result = pd.merge(result_water,result_rh,on=['地区','图像日期','东经','北纬','fire'])

    name_mapping = {

        '图像日期': 'time',
        '东经': 'lon',
        '北纬': 'lat'

    }
    final_result.rename(columns=name_mapping, inplace=True)
    final_result['time'] = pd.to_datetime(final_result['time'])
    final_result['date'] = final_result['time'].dt.date
    final_result['date'] = pd.to_datetime(final_result['date'])
    print(final_result)
    final_result = final_result.drop(columns=['地区', 'time','fire'])
    print('nc插值结果：',final_result)
#=========================================================
    name_mapping = {
        '地区': 'area',
        '图像日期': 'time',
        '东经': 'lon',
        '北纬': 'lat',
        'fire': 'fire'
    }
    fire.rename(columns=name_mapping, inplace=True)

    # 转换时间列并添加时间特征
    fire['time'] = pd.to_datetime(fire['time'])
    fire['year'] = fire['time'].dt.year
    fire['month'] = fire['time'].dt.month
    fire['day'] = fire['time'].dt.day
    # fire['hour'] = fire['time'].dt.hour
    fire['date'] = fire['time'].dt.date
    fire['date'] = pd.to_datetime(fire['date'])

    sta = pd.read_csv('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/原始数据/dataset.csv')

    print(sta.head())
    sta['date'] = pd.to_datetime(sta[['year', 'month', 'day']])

    merged_data = pd.merge(sta, fire, left_on='date', right_on='date', how='inner')
    merged_data['distance'] = np.sqrt((merged_data['lon_x'] - merged_data['lon_y']) ** 2 +
                                      (merged_data['lat_x'] - merged_data['lat_y']) ** 2)

    merged_data = merged_data.sort_values(by=['date', 'area', 'distance'])
    merged_data = merged_data.drop_duplicates()
    column_names = ['Alti', 'TEM_Max', 'TEM_Min',
                    'RHU_Min', 'PRE_Time_2020', 'Snow_Depth', 'WIN_S_Max',
                    'yth1', 'yth10', 'yth100', 'yth1000', 'kb', 'erc', 'sc', 'bi', 'p',
                    'FFMC', 'DMC', 'DC', 'FWI', 'ISI', 'BUI', 'DSR', 'FFDI', 'ic']
    numeric_cols = merged_data[column_names].select_dtypes(include=[np.number]).columns
    merged_data = merged_data[(merged_data[numeric_cols] <= 9000).all(axis=1)]
    merged_data = merged_data.groupby(['date','lon_y','lat_y']).head(200).reset_index(drop=True)

    column_names_list = merged_data.columns.tolist()
    print(column_names_list)

    final_interpolated_results = None

    # 循环遍历每个特征并进行插值
    for column_name in column_names:

        grouped_data = list(merged_data.groupby(['date','lon_y','lat_y']))
        args_list = [(group, column_name) for _, group in grouped_data]

        with Pool(processes=10) as pool:
            interpolated_results_list = pool.starmap(print_and_perform_kriging, args_list)

        interpolated_results = pd.concat(interpolated_results_list)

        # # 对当前特征进行插值
        # interpolated_results = merged_data.groupby('date').apply(
        #     lambda group: print_and_perform_kriging(group, column_name))
        # print(interpolated_results)
        if interpolated_results.empty:
            print(f"跳过空的插值结果，列名为 {column_name}")
            continue

        # 重置索引
        interpolated_results.reset_index(drop=True, inplace=True)
        print(interpolated_results)
        # 将插值结果合并到最终的DataFrame中
        if final_interpolated_results is None:
            final_interpolated_results = interpolated_results
        else:
            common_columns = ['date', 'lon', 'lat', 'year', 'month', 'day', 'area', 'fire']
            final_interpolated_results = final_interpolated_results.merge(interpolated_results, on=common_columns)
    final_interpolated_results = pd.merge(final_interpolated_results,final_result,on=['date','lon','lat'])
    # 将最终插值结果保存到CSV文件中
    final_interpolated_results.to_csv('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/train_dataset/interpolated_results_west.csv')
    # final_interpolated_results.to_csv('/public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/dataset/train_dataset/interpolated_results_north.csv')

