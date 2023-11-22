import pandas as pd
from datetime import timedelta
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import warnings
from pandas.errors import DtypeWarning
from scipy.spatial import cKDTree
import numpy as np
from pykrige.ok import OrdinaryKriging
warnings.simplefilter(action='ignore', category=DtypeWarning)
warnings.filterwarnings('ignore')



def perform_kriging_batchwise(group, column_name):
    # 从数据组中提取经度、纬度和目标特征值
    lon = group['lon_x'].values
    lat = group['lat_x'].values
    target_values = group[column_name].values
    
    # 创建一个网格，用于进行数据插值
    grid_lon = np.linspace(min(lon), max(lon), 50)
    grid_lat = np.linspace(min(lat), max(lat), 50)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # 执行普通克里金插值
    ok = OrdinaryKriging(
        lon, lat, target_values, variogram_model='linear',
        verbose=False, enable_plotting=False
    )
    z, ss = ok.execute('grid', grid_lon, grid_lat)
    
    # 创建一个DataFrame用于保存插值结果
    interpolated_df = pd.DataFrame({
        'date': group['date'].iloc[0],  # 假设整个组的日期是相同的
        'lon': grid_lon.ravel(),
        'lat': grid_lat.ravel(),
        column_name: z.ravel()
    })
    
    return interpolated_df




fire = pd.read_excel('fire2010-2017.xls')
fire['fire'] = 1
# 时间序列扩充
new_rows_list = []

# 遍历所有唯一的火灾日期
for fire_date in fire['图像日期'].unique():
    fire_rows = fire[fire['图像日期'] == fire_date]
    
    # 检查前两天和后一天的窗口
    for offset in range(-10, 10):
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
# extended_fire_data.to_csv(r'E:\dataset\fire2010-2017_new.csv')

provinces = ['辽宁省', '黑龙江省', '吉林省']
fire = extended_fire_data[extended_fire_data['地区'].isin(provinces)].copy()

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
fire['hour'] = fire['time'].dt.hour
fire['date'] = fire['time'].dt.date
fire['date'] = pd.to_datetime(fire['date'])



sta = pd.read_csv('dataset.csv')
print(sta.head())
sta['date'] = pd.to_datetime(sta[['year', 'month', 'day']])

merged_data = pd.merge(sta, fire, left_on='date', right_on='date', how='inner')
merged_data['distance'] = np.sqrt((merged_data['lon_x'] - merged_data['lon_y'])**2 + 
                                  (merged_data['lat_x'] - merged_data['lat_y'])**2)


merged_data = merged_data.sort_values(by=['date','area', 'distance']).drop_duplicates(subset='date', keep='first')

column_names_list = merged_data.columns.tolist()
print(column_names_list)
column_names = [ 'Alti', 'TEM_Max', 'TEM_Min', 
                'RHU_Min', 'PRE_Time_2020', 'Snow_Depth', 'WIN_S_Max',  
                'yth1', 'yth10', 'yth100', 'yth1000', 'kb', 'erc', 'sc', 'bi', 'p', 
                'FFMC', 'DMC', 'DC', 'FWI', 'ISI', 'BUI', 'DSR', 'FFDI','ic']


final_interpolated_results = None

# 循环遍历每个特征并进行插值
for column_name in column_names:
    # 对当前特征进行插值
    interpolated_results = merged_data.groupby('date').apply(lambda group: perform_kriging_batchwise(group, column_name))
    
    # 重置索引
    interpolated_results.reset_index(drop=True, inplace=True)
    
    # 将插值结果合并到最终的DataFrame中
    if final_interpolated_results is None:
        final_interpolated_results = interpolated_results
    else:
        common_columns = ['date', 'lon', 'lat']
        final_interpolated_results = final_interpolated_results.merge(interpolated_results, on=common_columns)
        
# 将最终插值结果保存到CSV文件中
final_interpolated_results.to_csv('final_interpolated_results.csv')