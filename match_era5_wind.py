import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


def match_era5_wind(processed_buoy_file_with_currents, era5_dir, output_dir):
    """
    Matches ERA5 reanalysis wind data with buoy trajectories that already contain
    HYCOM current data. It then performs crucial feature engineering on the wind data.

    This function builds upon the previous step and performs the following tasks:
    1.  Loads the buoy trajectories that have been matched with HYCOM currents.
    2.  Dynamically determines the required time range from the buoy data.
    3.  Loads only the relevant time slice from the ERA5 wind data files.
    4.  For each point in each trajectory, it performs spatio-temporal interpolation
        on the ERA5 dataset to find the corresponding 10-meter wind components (u10, v10).
    5.  ***KEY FEATURE ENGINEERING***:
        a. Calculates the wind speed (magnitude).
        b. Encodes the periodic wind direction into two continuous features: sine and cosine components.
    6.  Saves the final, fully enriched trajectories to a new pickle file.

    Args:
        processed_buoy_file_with_currents (str): Path to the 'trajectories_with_currents.pkl' file.
        era5_dir (str): Directory containing the yearly ERA5 NetCDF/GRIB files.
        output_dir (str): Directory to save the final processed file.
    """
    print("--- 开始匹配ERA5风场数据并进行特征工程 ---")

    # --- 1. 加载已匹配海流的浮标轨迹 ---
    print(f"步骤 1/7: 加载浮标数据从: {processed_buoy_file_with_currents}")
    try:
        with open(processed_buoy_file_with_currents, 'rb') as f:
            trajectories_with_currents = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{processed_buoy_file_with_currents}'")
        return
    if not trajectories_with_currents:
        print("错误: 加载的浮标轨迹列表为空，无法继续。")
        return
    print(f"加载了 {len(trajectories_with_currents)} 段已匹配海流的轨迹。")

    # --- 2. 从浮标数据中动态获取所需时间范围 ---
    print("步骤 2/7: 动态计算所需的时间范围...")
    all_times = pd.concat([traj['time'] for traj in trajectories_with_currents])
    min_time = all_times.min() - pd.Timedelta(days=1)  # Add a buffer for interpolation
    max_time = all_times.max() + pd.Timedelta(days=1)  # Add a buffer for interpolation
    print(f"所需数据时间范围: 从 {min_time.strftime('%Y-%m-%d')} 到 {max_time.strftime('%Y-%m-%d')}")

    # --- 3. 加载ERA5风场数据 (仅加载所需时间段) ---
    # Find either .grib or .nc files
    era5_files = sorted(glob.glob(os.path.join(era5_dir, '*.grib')))
    if not era5_files:
        era5_files = sorted(glob.glob(os.path.join(era5_dir, '*.nc')))

    if not era5_files:
        print(f"错误: 在目录 '{era5_dir}' 中未找到ERA5 .grib 或 .nc 文件。")
        return

    print(f"步骤 3/7: 使用 xarray.open_mfdataset 加载ERA5数据...")
    print(f"找到 {len(era5_files)} 个ERA5文件。将仅加载时间范围内的数据。")

    def select_time_range(ds):
        return ds.sel(time=slice(min_time, max_time))

    try:
        # Let xarray automatically choose the engine. It will pick 'cfgrib' for .grib
        # and 'netcdf4' for .nc, if they are installed.
        ds_era5_raw = xr.open_mfdataset(
            era5_files,
            combine='by_coords',
            preprocess=select_time_range
        )
    except ValueError as e:
        if "cfgrib" in str(e):  # A more general check for cfgrib related errors
            print("\n" + "=" * 80)
            print("致命错误: 使用 'cfgrib' 引擎读取GRIB文件失败。")
            print("这通常是由于环境配置问题导致的。")
            print("请尝试以下解决方案:")
            print("1. (推荐) 创建一个纯净的Conda环境并重新安装所有库。")
            print("   命令: conda create --name grib_env python=3.9 -y")
            print("         conda activate grib_env")
            print("         conda install -c conda-forge xarray pandas cfgrib netcdf4")
            print("2. (备选) 将您的GRIB文件手动转换为NetCDF格式后再运行此脚本。")
            print("   命令: grib_to_netcdf -o <output_file>.nc <input_file>.grib")
            print("=" * 80 + "\n")
            return
        else:
            print(f"读取文件时发生未预料的ValueError: {e}")
            raise
    except Exception as e:
        print(f"读取数据文件时发生未知错误: {e}")
        return

    # 重命名变量以便于访问
    rename_dict = {
        '10u': 'u10',
        '10v': 'v10',
        'u10': 'u10',
        'v10': 'v10'
    }
    actual_rename_dict = {k: v for k, v in rename_dict.items() if k in ds_era5_raw.variables}
    ds_era5 = ds_era5_raw.rename(actual_rename_dict)

    if 'lon' in ds_era5.coords and ds_era5.lon.min() < 0:
        ds_era5['lon'] = (ds_era5['lon'] + 360) % 360
    ds_era5 = ds_era5.sortby('lon')

    print("ERA5 数据集在指定时间范围内加载并预处理完成。")
    print(f"加载后的数据集时间范围: {ds_era5.time.min().values} to {ds_era5.time.max().values}")

    # --- 4, 5, 6. 迭代、插值与特征工程 ---
    print("步骤 4-6/7: 迭代所有轨迹，进行插值和风场特征工程...")
    fully_enriched_trajectories = []
    for traj_df in tqdm(trajectories_with_currents, desc="匹配风场数据中"):
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")
        lons_360 = (lons + 360) % 360

        try:
            interpolated_wind = ds_era5[['u10', 'v10']].interp(
                lat=lats,
                lon=lons_360,
                time=times,
                method="linear"
            )

            u10 = interpolated_wind['u10'].values
            v10 = interpolated_wind['v10'].values
            traj_df['era5_u10'] = u10
            traj_df['era5_v10'] = v10

            # --- 核心特征工程 ---
            wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
            traj_df['era5_wind_speed'] = wind_speed

            wind_angle_rad = np.arctan2(v10, u10)
            traj_df['era5_wind_dir_sin'] = np.sin(wind_angle_rad)
            traj_df['era5_wind_dir_cos'] = np.cos(wind_angle_rad)

            traj_df.dropna(subset=['era5_u10', 'era5_v10'], inplace=True)

            if len(traj_df) > 1:
                fully_enriched_trajectories.append(traj_df)

        except Exception as e:
            print(f"\n警告: 处理某段轨迹时发生插值错误: {e}")
            print("该段轨迹将被跳过。")
            continue

    print(f"处理完成。剩余 {len(fully_enriched_trajectories)} 段轨迹拥有匹配的风场数据。")

    # --- 7. 保存最终结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'trajectories_with_currents_and_wind.pkl')
    print(f"\n步骤 7/7: 将包含所有特征的最终结果保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(fully_enriched_trajectories, f)

    print("--- ERA5风场数据匹配与特征工程全部完成！---")

    if fully_enriched_trajectories:
        print(f"\n最终产出是一个Python列表，其中包含 {len(fully_enriched_trajectories)} 个pandas.DataFrame。")
        print("每个DataFrame现在都包含了浮标、HYCOM海流、ERA5风场以及风场衍生特征。")
        print("\n第一个轨迹的头部数据示例:")
        print(fully_enriched_trajectories[0].head())
        print("\n查看新增的列名:")
        print(fully_enriched_trajectories[0].columns)
    else:
        print("\n警告: 没有轨迹成功匹配风场数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    PROCESSED_BUOY_FILE_WITH_CURRENTS = './processed_data/trajectories_with_currents.pkl'
    ERA5_DATA_DIRECTORY = './reanalysis/wind'
    OUTPUT_DIRECTORY = './processed_data'

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_CURRENTS):
        print(f"错误: 输入文件 '{PROCESSED_BUOY_FILE_WITH_CURRENTS}' 不存在。请先运行海流匹配脚本。")
    elif not os.path.exists(ERA5_DATA_DIRECTORY):
        print(f"错误: ERA5数据目录 '{ERA5_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wind(PROCESSED_BUOY_FILE_WITH_CURRENTS, ERA5_DATA_DIRECTORY, OUTPUT_DIRECTORY)

