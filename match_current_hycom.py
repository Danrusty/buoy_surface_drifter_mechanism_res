import xarray as xr
import pandas as pd
import numpy as np
from joblib import parallel,delayed
import os
import pickle
import glob
from tqdm import tqdm


def match_hycom_currents(processed_buoy_file, hycom_dir, output_dir):
    """
    Matches HYCOM reanalysis current data with preprocessed buoy trajectories.

    This function performs two main tasks:
    1. Upsamples the 6-hourly buoy trajectories to a 1-hour resolution.
    2. For each point in the new 1-hourly trajectories, it performs spatio-temporal
       interpolation on the HYCOM dataset to find the corresponding background
       sea water velocity (u and v components).

    Args:
        processed_buoy_file (str): Path to the 'processed_undrogued_buoy_trajectories.pkl' file.
        hycom_dir (str): Directory containing the monthly HYCOM NetCDF files.
        output_dir (str): Directory to save the final processed file.
    """
    print("--- 开始匹配HYCOM海流数据 ---")

    # --- 1. 加载预处理过的浮标轨迹 ---
    print(f"步骤 1/6: 加载浮标数据从: {processed_buoy_file}")
    try:
        with open(processed_buoy_file, 'rb') as f:
            buoy_trajectories = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 浮标数据文件未找到 at '{processed_buoy_file}'")
        return
    print(f"加载了 {len(buoy_trajectories)} 段连续的无水帆浮标轨迹。")

    # --- 2. 上采样浮标轨迹至1小时间隔 ---
    print("步骤 2/6: 将所有轨迹上采样至1小时间隔...")
    hourly_trajectories = []
    for traj in tqdm(buoy_trajectories, desc="上采样轨迹中"):
        if traj.empty:
            continue

        # --- FIX: Robust upsampling logic ---
        # 1. Store the non-numeric metadata first.
        buoy_id = traj['ID'].iloc[0]

        # 2. Set index for resampling.
        traj_indexed = traj.set_index('time')

        # 3. Select ONLY numeric columns for interpolation to avoid FutureWarning.
        numeric_cols = traj_indexed.select_dtypes(include=np.number)
        traj_hourly_numeric = numeric_cols.resample('h').interpolate(method='linear')

        # 4. Create the final hourly dataframe and re-assign the ID.
        traj_hourly = traj_hourly_numeric.reset_index()
        traj_hourly['ID'] = buoy_id

        # 5. Reorder columns to maintain a consistent structure.
        # Get original column order, ensure 'ID' is first.
        original_cols = traj.columns.tolist()
        if 'ID' in original_cols:
            original_cols.insert(0, original_cols.pop(original_cols.index('ID')))

        # Filter for columns that actually exist in the new dataframe
        final_cols_order = [col for col in original_cols if col in traj_hourly.columns]
        traj_hourly = traj_hourly[final_cols_order]

        # If interpolation results in less than 2 points, discard the trajectory.
        if len(traj_hourly) > 1:
            hourly_trajectories.append(traj_hourly)

    print(f"上采样完成。现在有 {len(hourly_trajectories)} 段1小时间隔的轨迹。")

    # --- 3. 加载HYCOM数据 ---
    # 使用 glob 找到所有 u 和 v 的文件
    u_files = sorted(glob.glob(os.path.join(hycom_dir, '*eastward*.nc')))
    v_files = sorted(glob.glob(os.path.join(hycom_dir, '*northward*.nc')))

    if not u_files or not v_files:
        print(f"错误: 在目录 '{hycom_dir}' 中未找到HYCOM NetCDF文件。")
        print("请确保文件名包含 'eastward' (u) 和 'northward' (v)。")
        return
    print(f"步骤 3/6: 使用 xarray.open_mfdataset 加载HYCOM数据...")
    print(f"找到 {len(u_files)} 个 U-分量文件和 {len(v_files)} 个 V-分量文件。")

    ds_u_raw = xr.open_mfdataset(u_files, combine='by_coords', parallel=True)
    ds_v_raw = xr.open_mfdataset(v_files, combine='by_coords', parallel=True)

    # --- FIX: Remove duplicate time entries BEFORE merging ---
    # Monthly files can have overlapping start/end dates. This creates duplicate time values
    # which can cause errors during merge or interpolation. We de-duplicate each dataset first.
    print("正在剔除U分量数据中的重复时间戳...")
    _, u_unique_indices = np.unique(ds_u_raw['time'], return_index=True)
    ds_u = ds_u_raw.isel(time=u_unique_indices)
    print(f"U分量有效时间维度大小为: {len(ds_u['time'])}")

    print("正在剔除V分量数据中的重复时间戳...")
    _, v_unique_indices = np.unique(ds_v_raw['time'], return_index=True)
    ds_v = ds_v_raw.isel(time=v_unique_indices)
    print(f"V分量有效时间维度大小为: {len(ds_v['time'])}")

    # Now merge the de-duplicated datasets
    print("正在合并已去重的数据集...")
    ds_hycom = xr.merge([ds_u, ds_v])
    # 移除深度维度，因为它只有一个值，可以简化插值
    ds_hycom = ds_hycom.squeeze()
    print("HYCOM 数据集加载并合并完成。")


    # --- 4, 5. 迭代、匹配和插值 ---
    print("步骤 4&5/6: 迭代所有轨迹并进行时空插值...")
    enriched_trajectories = []
    for traj_df in tqdm(hourly_trajectories, desc="插值海流数据中"):
        # 准备插值所需的坐标数组
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")

        # --- 关键: 经度坐标转换 ---
        # 浮标经度是 -180 到 180, HYCOM 是 0 到 360
        lons_360 = (lons + 360) % 360

        try:
            # 使用 xarray 的高级插值功能
            # 一次性对所有点进行插值，效率很高
            interpolated_currents = ds_hycom[['water_u','water_v']].interp(
                lat=lats,
                lon=lons_360,
                time=times,
                method="linear",  # 对所有维度使用线性插值
               # kwargs={"fill_value": "extrapolate"}  # 允许在边界进行外插
            )

            # 将插值结果添加回 DataFrame
            traj_df['hycom_u'] = interpolated_currents['water_u'].values
            traj_df['hycom_v'] = interpolated_currents['water_v'].values

            # 删除插值失败（结果为NaN）的行
            traj_df.dropna(subset=['hycom_u', 'hycom_v'], inplace=True)

            # 只有当轨迹仍然有足够数据时才保留
            if len(traj_df) > 1:
                enriched_trajectories.append(traj_df)

        except Exception as e:
            print(f"\n警告: 处理某段轨迹时发生插值错误: {e}")
            print("该段轨迹将被跳过。")
            continue

    print(f"插值完成。剩余 {len(enriched_trajectories)} 段轨迹拥有匹配的海流数据。")

    # --- 6. 保存结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'trajectories_with_currents.pkl')
    print(f"\n步骤 6/6: 将富含海流数据的结果保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(enriched_trajectories, f)

    print("--- HYCOM海流数据匹配完成！---")

    if enriched_trajectories:
        print(f"\n最终产出是一个Python列表，其中包含 {len(enriched_trajectories)} 个pandas.DataFrame。")
        print("每个DataFrame代表一个【无水帆】浮标的一段【1小时间隔】【连续】轨迹，并包含了匹配的HYCOM海流速度。")
        print("\n第一个轨迹的头部数据示例:")
        print(enriched_trajectories[0].head())
    else:
        print("\n警告: 没有轨迹成功匹配海流数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 1. 上一步生成的预处理浮标文件
    PROCESSED_BUOY_FILE = './processed_data/processed_undrogued_buoy_trajectories.pkl'

    # 2. 存放所有HYCOM NetCDF文件的目录
    #    例如: 'D:/reanalysis_data/hycom/'
    HYCOM_DATA_DIRECTORY = r'C:\Users\dan\DISK_D\post_doc_research\buoy2025\reanalysis\current'

    # 3. 输出目录
    OUTPUT_DIRECTORY = './processed_data'

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE):
        print(f"错误: 输入的浮标文件 '{PROCESSED_BUOY_FILE}' 不存在。请先运行第一步预处理脚本。")
    elif not os.path.exists(HYCOM_DATA_DIRECTORY):
        print(f"错误: HYCOM数据目录 '{HYCOM_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_hycom_currents(PROCESSED_BUOY_FILE, HYCOM_DATA_DIRECTORY, OUTPUT_DIRECTORY)