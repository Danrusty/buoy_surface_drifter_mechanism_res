import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


def match_era5_wave(processed_buoy_file_with_wind, era5_wave_dir, output_dir):
    """
    Final data processing step: Matches ERA5 reanalysis wave data with trajectories
    that already contain current and wind data. It also performs feature engineering on wave direction.

    This function completes the feature set by performing the following tasks:
    1.  Loads the buoy trajectories enriched with HYCOM currents and ERA5 wind.
    2.  Dynamically determines the required time range from the buoy data.
    3.  Loads the relevant time slice from the yearly ERA5 wave data NetCDF files.
    4.  Performs spatio-temporal interpolation to find the corresponding wave parameters
        (swh, mwp, mwd) for each point in each trajectory.
    5.  ***KEY FEATURE ENGINEERING***: Encodes the periodic mean wave direction ('mwd')
        into two continuous features: sine and cosine components.
    6.  Saves the final, fully enriched trajectories, ready for model training.

    Args:
        processed_buoy_file_with_wind (str): Path to the 'trajectories_with_currents_and_wind.pkl' file.
        era5_wave_dir (str): Directory containing the yearly ERA5 wave NetCDF files.
        output_dir (str): Directory to save the final, complete dataset.
    """
    print("--- 开始匹配ERA5波浪数据并进行特征工程 (最终数据准备步骤) ---")

    # --- 1. 加载已匹配海流和风场的浮标轨迹 ---
    print(f"步骤 1/7: 加载浮标数据从: {processed_buoy_file_with_wind}")
    try:
        with open(processed_buoy_file_with_wind, 'rb') as f:
            trajectories_with_wind = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{processed_buoy_file_with_wind}'")
        return
    if not trajectories_with_wind:
        print("错误: 加载的浮标轨迹列表为空，无法继续。")
        return
    print(f"加载了 {len(trajectories_with_wind)} 段已匹配海流和风场的轨迹。")

    # --- 2. 动态获取所需时间范围 ---
    print("步骤 2/7: 动态计算所需的时间范围...")
    all_times = pd.concat([traj['time'] for traj in trajectories_with_wind])
    min_time = all_times.min() - pd.Timedelta(days=1)
    max_time = all_times.max() + pd.Timedelta(days=1)
    print(f"所需数据时间范围: 从 {min_time.strftime('%Y-%m-%d')} 到 {max_time.strftime('%Y-%m-%d')}")

    # --- 3. 加载ERA5波浪数据 ---
    era5_wave_files = sorted(glob.glob(os.path.join(era5_wave_dir, '*.nc')))
    if not era5_wave_files:
        print(f"错误: 在目录 '{era5_wave_dir}' 中未找到ERA5波浪 .nc 文件。")
        return

    print(f"步骤 3/7: 使用 xarray.open_mfdataset 加载ERA5波浪数据...")
    print(f"找到 {len(era5_wave_files)} 个ERA5波浪文件。")

    def select_time_range_and_rename(ds):
        # Rename 'valid_time' to 'time' to match other datasets and avoid conflicts
        if 'valid_time' in ds.coords:
            ds = ds.rename({'valid_time': 'time'})
        return ds.sel(time=slice(min_time, max_time))

    ds_era5_wave_raw = xr.open_mfdataset(
        era5_wave_files,
        combine='by_coords',
        preprocess=select_time_range_and_rename
    )

    # Standardize coordinate names
    ds_era5_wave = ds_era5_wave_raw.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Ensure longitude is in 0-360 range
    if 'lon' in ds_era5_wave.coords and ds_era5_wave.lon.min() < 0:
        ds_era5_wave['lon'] = (ds_era5_wave['lon'] + 360) % 360
    ds_era5_wave = ds_era5_wave.sortby('lon')

    print("ERA5 波浪数据集在指定时间范围内加载并预处理完成。")

    # --- 4, 5, 6. 迭代、插值与特征工程 ---
    print("步骤 4-6/7: 迭代所有轨迹，进行插值和波浪特征工程...")
    final_trajectories = []
    for traj_df in tqdm(trajectories_with_wind, desc="匹配波浪数据中"):
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")
        lons_360 = (lons + 360) % 360

        try:
            interpolated_wave = ds_era5_wave[['swh', 'mwp', 'mwd']].interp(
                lat=lats,
                lon=lons_360,
                time=times,
                method="linear"
            )

            # Add wave height and period directly
            traj_df['era5_swh'] = interpolated_wave['swh'].values
            traj_df['era5_mwp'] = interpolated_wave['mwp'].values

            # --- 核心特征工程: 处理周期性的波浪方向 ---
            mwd_deg = interpolated_wave['mwd'].values
            # Convert degrees to radians for trigonometric functions
            mwd_rad = np.deg2rad(mwd_deg)
            traj_df['era5_wave_dir_sin'] = np.sin(mwd_rad)
            traj_df['era5_wave_dir_cos'] = np.cos(mwd_rad)

            # Drop rows where interpolation failed
            traj_df.dropna(subset=['era5_swh', 'era5_mwp', 'era5_wave_dir_sin'], inplace=True)

            if len(traj_df) > 1:
                final_trajectories.append(traj_df)

        except Exception as e:
            print(f"\n警告: 处理某段轨迹时发生插值错误: {e}")
            print("该段轨迹将被跳过。")
            continue

    print(f"处理完成。剩余 {len(final_trajectories)} 段轨迹拥有完整的环境数据。")

    # --- 7. 保存最终结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'trajectories_with_all_features.pkl')
    print(f"\n步骤 7/7: 将包含所有特征的最终数据集保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(final_trajectories, f)

    print("\n" + "=" * 80)
    print("--- 所有数据预处理和特征工程步骤已全部完成！---")
    print("=" * 80)

    if final_trajectories:
        print(f"\n最终产出文件 '{output_filename}' 是一个Python列表。")
        print("列表中的每个DataFrame都包含了构建深度学习模型所需的全部输入特征:")
        print("  - 浮标观测数据 (ID, time, lat, lon, ve, vn)")
        print("  - HYCOM 背景海流 (hycom_u, hycom_v)")
        print("  - ERA5 背景风场 (era5_u10, era5_v10)及其衍生特征 (speed, dir_sin, dir_cos)")
        print("  - ERA5 背景波浪 (era5_swh, era5_mwp)及其衍生特征 (dir_sin, dir_cos)")
        print("\n第一个轨迹的头部数据示例:")
        print(final_trajectories[0].head())
        print("\n最终数据集的完整列名:")
        print(final_trajectories[0].columns.tolist())
    else:
        print("\n警告: 没有轨迹成功匹配波浪数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 1. 上一步生成的、已匹配海流和风场的文件
    PROCESSED_BUOY_FILE_WITH_WIND = './processed_data/trajectories_with_currents_and_wind.pkl'

    # 2. 存放所有ERA5波浪NetCDF文件的目录
    #    例如: 'D:/reanalysis_data/era5_wave/'
    ERA5_WAVE_DATA_DIRECTORY = './reanalysis/wave'

    # 3. 输出目录
    OUTPUT_DIRECTORY = './processed_data'

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_WIND):
        print(f"错误: 输入文件 '{PROCESSED_BUOY_FILE_WITH_WIND}' 不存在。请先运行风场匹配脚本。")
    elif not os.path.exists(ERA5_WAVE_DATA_DIRECTORY):
        print(f"错误: ERA5波浪数据目录 '{ERA5_WAVE_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wave(PROCESSED_BUOY_FILE_WITH_WIND, ERA5_WAVE_DATA_DIRECTORY, OUTPUT_DIRECTORY)
