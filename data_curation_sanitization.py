import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from typing import List

# --- 1. 配置与设置 ---
# ==============================================================================
# 使用与主脚本相同的日志配置
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"data_analysis_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# --- 输入/输出文件配置 ---
INPUT_DATA_PATH = './processed_data/trajectories_with_all_features.pkl'
OUTPUT_SANITIZED_PATH = './processed_data/trajectories_sanitized.pkl'
RESULTS_OUTPUT_DIR = './results/data_analysis'
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# --- 清洗与过滤参数 ---
MAX_GAP_HOURS = 24  # 定义触发分割的最大时间间断（小时）
MIN_TRAJ_LENGTH  = 72  # 定义保留一条子轨迹所需的最小长度 (SEQUENCE_LENGTH + 1)


# ==============================================================================

def analyze_and_sanitize(input_path: str, output_path: str, max_gap_hours: int, min_length: int):
    """
    加载完整的轨迹数据集，对其进行质量分析，分割或清洗不合格数据，并保存净化后的版本。
    """
    logging.info(f"--- 启动数据分析与清洗流程 ---")
    logging.info(f"读取输入文件: {input_path}")

    try:
        with open(input_path, 'rb') as f:
            all_trajectories = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"错误: 输入数据文件未找到！请确保 '{input_path}' 存在。")
        return

    initial_traj_count = len(all_trajectories)
    logging.info(f"成功加载 {initial_traj_count} 条原始轨迹。")

    # --- 步骤 1: 分析原始数据质量 (此步骤保持不变) ---
    # ... (代码与之前版本相同，为了简洁省略)
    logging.info("\n--- 步骤 1: 分析原始数据质量 ---")
    lengths = [len(t) for t in all_trajectories]
    # 确保中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 6));
    plt.hist(lengths, bins=50, alpha=0.7);
    plt.title('原始轨迹长度分布')
    plt.xlabel('数据点数量');
    plt.ylabel('轨迹数量');
    plt.grid(True);
    plt.yscale('log')
    plt.savefig(os.path.join(RESULTS_OUTPUT_DIR, 'original_length_distribution.png'));
    plt.close()
    logging.info(f"原始轨迹长度分布图已保存。")

    # --- 步骤 2: 清洗与分割数据 ---
    logging.info(f"\n--- 步骤 2: 清洗与分割数据 ---")
    logging.info(f"分割阈值: > {max_gap_hours} 小时, 最短保留长度: > {min_length - 1} 个点")

    sanitized_trajectories = []
    max_gap_threshold = pd.Timedelta(hours=max_gap_hours)

    for i, traj in enumerate(all_trajectories):
        if traj.empty or 'time' not in traj.columns or len(traj) < min_length:
            continue

        time_diffs = traj['time'].diff()
        split_indices = time_diffs[time_diffs > max_gap_threshold].index.tolist()

        if not split_indices:
            # 如果没有间断，且长度合格，直接保留整条轨迹
            sanitized_trajectories.append(traj.reset_index(drop=True))
        else:
            # 如果有间断，则进行分割
            logging.warning(f"轨迹 {i} 存在间断点，正在尝试分割...")
            last_split_idx = 0
            # 添加轨迹的起始点
            split_points = [0] + split_indices + [len(traj)]

            for j in range(len(split_points) - 1):
                start_idx = split_points[j]
                # 在间断点之后开始新的轨迹
                if j > 0: start_idx = split_points[j]

                end_idx = split_points[j + 1]
                sub_traj = traj.iloc[start_idx:end_idx]

                # 增加最小长度过滤器
                if len(sub_traj) >= min_length:
                    sanitized_trajectories.append(sub_traj.reset_index(drop=True))
                    logging.info(f"  -> 从轨迹 {i} 中成功分割并保留了一段长度为 {len(sub_traj)} 的子轨迹。")
                else:
                    logging.warning(f"  -> 从轨迹 {i} 中分割的一段子轨迹因长度 ({len(sub_traj)}) 过短而被丢弃。")

    final_traj_count = len(sanitized_trajectories)
    logging.info(
        f"数据清洗与分割完成。从 {initial_traj_count} 条原始轨迹中，共生成了 {final_traj_count} 条干净、连续且长度足够的轨迹。")

    # --- 步骤 3: 分析清洗后的数据质量 ---
    logging.info("\n--- 步骤 3: 分析清洗后的数据质量 ---")
    final_lengths = [len(t) for t in sanitized_trajectories]
    plt.figure(figsize=(12, 6))
    plt.hist(final_lengths, bins=50, alpha=0.7, color='green')
    plt.title('清洗后的轨迹长度分布')
    plt.xlabel('轨迹中的数据点数量');
    plt.ylabel('轨迹数量');
    plt.grid(True);
    plt.yscale('log')
    plt.savefig(os.path.join(RESULTS_OUTPUT_DIR, 'sanitized_length_distribution.png'));
    plt.close()
    logging.info(
        f"清洗后的轨迹长度分布图已保存。平均长度: {np.mean(final_lengths):.1f}, 中位数: {np.median(final_lengths):.1f}")

    # --- 步骤 4: 保存净化后的数据 ---
    logging.info(f"\n--- 步骤 4: 保存净化后的数据集 ---")
    logging.info(f"将 {final_traj_count} 条干净的轨迹保存到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(sanitized_trajectories, f)

    logging.info("--- 数据分析与清洗流程全部完成 ---")


if __name__ == '__main__':
    analyze_and_sanitize(INPUT_DATA_PATH, OUTPUT_SANITIZED_PATH, MAX_GAP_HOURS, MIN_TRAJ_LENGTH)

