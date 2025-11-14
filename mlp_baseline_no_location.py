import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Any

# --- 1. 配置与设置 ---
# ==============================================================================
# 设置日志、模型和结果的保存目录
LOG_DIR = './logs'
MODEL_OUTPUT_DIR = './models'
RESULTS_OUTPUT_DIR = './results'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# 设置日志记录
log_filename = f"mlp_training_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# 获取根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 创建格式化器
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# 创建并配置一个文件处理器，强制使用UTF-8编码，以避免在Windows上出现编码错误
file_handler = logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8')
file_handler.setFormatter(formatter)
# 创建并配置一个流处理器（用于控制台输出）
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
# 将处理器添加到日志记录器
# 避免重复添加处理器
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

# --- 全局常量与超参数 ---
FINAL_DATA_PATH = './processed_data/trajectories_with_all_features.pkl'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数
LEARNING_RATE = 0.001
BATCH_SIZE = 1024
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
RANDOM_SEED = 42

# 模型架构参数
MLP_INPUT_SIZE = 7  # 根据特征数量确定
MLP_OUTPUT_SIZE = 2  # 预测 delta_u 和 delta_v
MLP_HIDDEN_LAYERS = [128, 256, 128]
MLP_DROPOUT_RATE = 0.1

# 设置随机种子以保证结果可复现
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================================================================


# --- 2. 数据加载与准备 ---
# ==============================================================================
def load_and_prepare_data(filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[None, None, None]:
    """
    加载最终处理好的数据，并准备用于MLP模型的特征和目标。

    Args:
        filepath (str): 数据文件路径 (.pkl)。

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | Tuple[None, None, None]:
        特征(X), 目标(y), 和包含轨迹ID的完整DataFrame。如果失败则返回None。
    """
    logging.info(f"--- 步骤 1: 从 '{filepath}' 加载数据 ---")
    try:
        with open(filepath, 'rb') as f:
            all_trajectories = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"错误: 数据文件未找到！请确保 '{filepath}' 存在。")
        return None, None, None

    feature_columns = [
        'era5_wind_speed', 'era5_wind_dir_sin', 'era5_wind_dir_cos',
        'era5_swh', 'era5_mwp', 'era5_wave_dir_sin', 'era5_wave_dir_cos'
    ]
    observation_columns = ['ve', 'vn', 'hycom_u', 'hycom_v']

    valid_trajectories = [
        traj for traj in all_trajectories
        if set(feature_columns + observation_columns).issubset(traj.columns)
    ]

    if not valid_trajectories:
        logging.error("错误: 没有任何有效的轨迹可用于训练。")
        return None, None, None

    logging.info(f"数据加载完成，共 {len(valid_trajectories)} 段有效轨迹。")
    logging.info("--- 步骤 2: 计算目标残差并准备数据集 ---")

    # 为每条轨迹分配唯一ID，用于后续的科学划分
    for i, traj_df in enumerate(valid_trajectories):
        traj_df['trajectory_id'] = i

    all_points_df = pd.concat(valid_trajectories, ignore_index=True)

    # 计算目标：漂流浮标观测速度与HYCOM模型速度的残差
    all_points_df['delta_u'] = all_points_df['ve'] - all_points_df['hycom_u']
    all_points_df['delta_v'] = all_points_df['vn'] - all_points_df['hycom_v']

    target_columns = ['delta_u', 'delta_v']
    X = all_points_df[feature_columns]
    y = all_points_df[target_columns]

    # 验证输入特征数量是否与模型配置一致
    global MLP_INPUT_SIZE
    MLP_INPUT_SIZE = len(feature_columns)

    logging.info(f"数据集准备完成，共 {len(X)} 个样本点。")
    logging.info(f"输入特征 ({MLP_INPUT_SIZE}个): {feature_columns}")
    logging.info(f"预测目标 ({MLP_OUTPUT_SIZE}个): {target_columns}")

    return X, y, all_points_df


def split_and_scale_data(X: pd.DataFrame, y: pd.DataFrame, df: pd.DataFrame) -> Tuple:
    """
    按轨迹ID划分数据集以防数据泄露，并对数据进行标准化。

    Args:
        X (pd.DataFrame): 特征数据。
        y (pd.DataFrame): 目标数据。
        df (pd.DataFrame): 包含 'trajectory_id' 的完整数据。

    Returns:
        Tuple: 包含划分和标准化后的数据集及标准化器的元组。
    """
    logging.info("--- 步骤 3: 按轨迹划分并标准化数据 ---")

    unique_traj_ids = df['trajectory_id'].unique()
    train_val_ids, test_ids = train_test_split(unique_traj_ids, test_size=0.15, random_state=RANDOM_SEED)
    train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1, random_state=RANDOM_SEED)  # 验证集占非测试集的10%

    X_train = X[df['trajectory_id'].isin(train_ids)]
    y_train = y[df['trajectory_id'].isin(train_ids)]
    X_val = X[df['trajectory_id'].isin(val_ids)]
    y_val = y[df['trajectory_id'].isin(val_ids)]
    X_test = X[df['trajectory_id'].isin(test_ids)]
    y_test = y[df['trajectory_id'].isin(test_ids)]

    logging.info("数据集划分完成:")
    logging.info(f"  - 训练集: {len(X_train)} 个样本 (来自 {len(train_ids)} 条轨迹)")
    logging.info(f"  - 验证集: {len(X_val)} 个样本 (来自 {len(val_ids)} 条轨迹)")
    logging.info(f"  - 测试集: {len(X_test)} 个样本 (来自 {len(test_ids)} 条轨迹)")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    X_val_scaled = x_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    X_test_scaled = x_scaler.transform(X_test)
    # y_test保持原始尺度，用于最终评估

    logging.info("数据标准化完成。")

    with open(os.path.join(MODEL_OUTPUT_DIR, 'mlp_x_scaler_no_loc.pkl'), 'wb') as f: pickle.dump(x_scaler, f)
    with open(os.path.join(MODEL_OUTPUT_DIR, 'mlp_y_scaler_no_loc.pkl'), 'wb') as f: pickle.dump(y_scaler, f)
    logging.info(f"标准化器已保存到 '{MODEL_OUTPUT_DIR}' 目录。")

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test, y_scaler


# ==============================================================================


# --- 3. 模型定义 ---
# ==============================================================================
class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_layers: List[int], dropout_rate: float):
        super(MLP, self).__init__()
        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        layers.append(nn.Linear(in_features, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ==============================================================================


# --- 4. 训练与评估辅助函数 ---
# ==============================================================================
def _run_inference_pass(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    [辅助函数] 在给定的数据集上运行一次前向传播（推理）。

    Args:
        model (nn.Module): 待评估的模型。
        loader (DataLoader): 数据加载器。
        device (torch.device): 计算设备。

    Returns:
        Tuple[np.ndarray, np.ndarray]: (预测结果, 真实目标)。
    """
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch.to(device))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, patience: int, device: torch.device) -> Dict[str, List[float]]:
    """
    训练模型，包含早停机制，并记录训练历史。

    Returns:
        Dict[str, List[float]]: 包含训练/验证损失和R²的历史记录字典。
    """
    logging.info("\n--- 步骤 4: 开始模型训练 ---")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_mlp_model_no_loc.pth')

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # --- 验证阶段 ---
        val_preds_scaled, val_targets_scaled = _run_inference_pass(model, val_loader, device)
        val_loss = np.mean((val_preds_scaled - val_targets_scaled) ** 2)
        val_r2 = r2_score(val_targets_scaled, val_preds_scaled)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)

        logging.info(
            f"Epoch [{epoch + 1:03d}/{epochs:03d}] | "
            f"训练损失: {epoch_train_loss:.6f} | "
            f"验证损失: {val_loss:.6f} | "
            f"验证 R²: {val_r2:.4f}"
        )

        # --- 早停与模型保存 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  -> 验证损失改善，已保存最佳模型到 '{best_model_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"\n连续 {patience} 个 epoch 验证损失未改善，触发早停。")
                break

    logging.info("--- 训练完成 ---")
    return history


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """计算 MAPE，注意：当真实值接近0时，该指标可能不稳定。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # 避免除以零
    denominator = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100


def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    """计算调整后的 R²。"""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) != 0 else r2


def evaluate_model(model: nn.Module, test_loader: DataLoader, y_scaler: StandardScaler,
                   device: torch.device, num_features: int) -> None:
    """
    在测试集上评估最终模型，并打印详细性能指标。
    """
    logging.info("\n--- 步骤 5: 在测试集上评估最终模型 ---")

    # 运行推理，获取标准化后的预测和原始尺度的目标
    preds_scaled, targets_original_np = _run_inference_pass(model, test_loader, device)

    # 将预测值反标准化回原始物理单位 (m/s)
    preds_original = y_scaler.inverse_transform(preds_scaled)

    # --- 计算各项指标 ---
    r2_u = r2_score(targets_original_np[:, 0], preds_original[:, 0])
    r2_v = r2_score(targets_original_np[:, 1], preds_original[:, 1])
    r2_overall = r2_score(targets_original_np, preds_original)
    adj_r2_overall = adjusted_r2_score(r2_overall, len(targets_original_np), num_features)
    mape_u = mean_absolute_percentage_error(targets_original_np[:, 0], preds_original[:, 0])
    mape_v = mean_absolute_percentage_error(targets_original_np[:, 1], preds_original[:, 1])
    mae = np.mean(np.abs(preds_original - targets_original_np))
    rmse = np.sqrt(np.mean((preds_original - targets_original_np) ** 2))

    logging.info("最终模型在测试集上的性能:")
    logging.info("  --- 误差指标 (越小越好) ---")
    logging.info(f"  - MAE (m/s): {mae:.4f}")
    logging.info(f"  - RMSE (m/s): {rmse:.4f}")
    logging.info(f"  - MAPE (U-分量, %): {mape_u:.2f}%")
    logging.info(f"  - MAPE (V-分量, %): {mape_v:.2f}%")
    logging.info("  --- 拟合优度指标 (越接近1越好) ---")
    logging.info(f"  - R² (U-分量): {r2_u:.4f}")
    logging.info(f"  - R² (V-分量): {r2_v:.4f}")
    logging.info(f"  - R² (总体): {r2_overall:.4f}")
    logging.info(f"  - Adjusted R² (总体): {adj_r2_overall:.4f}")


def plot_history(history: Dict[str, List[float]]) -> None:
    """绘制训练过程中的损失和 R² 曲线图。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(history['train_loss'], label='训练损失 (Training Loss)', color='tab:blue')
    ax1.plot(history['val_loss'], label='验证损失 (Validation Loss)', color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('均方误差损失 (MSE Loss)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(history['val_r2'], label='验证 $R^2$ (Validation $R^2$)', color='tab:green', linestyle='--')
    ax2.set_ylabel('决定系数 $R^2$', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    ax1.set_title('MLP模型训练过程 (无位置特征)', fontsize=14, pad=20)
    fig.tight_layout(pad=2.0)

    plot_path = os.path.join(RESULTS_OUTPUT_DIR, 'mlp_loss_r2_curve_no_loc.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"\n训练历史曲线图已保存到: '{plot_path}'")
    plt.close()


# ==============================================================================


# --- 5. 主执行流程 ---
# ==============================================================================
def main():
    logging.info(f"--- 启动训练，将使用设备: {DEVICE} ---")

    X, y, df_all = load_and_prepare_data(FINAL_DATA_PATH)
    if X is None:
        logging.error("数据加载失败，程序终止。")
        return

    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test, y_scaler = split_and_scale_data(X, y, df_all)

    # 创建DataLoader
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train_s).float()),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_s).float(), torch.from_numpy(y_val_s).float()),
                            batch_size=BATCH_SIZE, shuffle=False)
    # 注意: 测试集的y (y_test) 未被标准化，这将在评估函数中处理
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_s).float(), torch.from_numpy(y_test.values).float()),
                             batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = MLP(
        input_size=MLP_INPUT_SIZE,
        output_size=MLP_OUTPUT_SIZE,
        hidden_layers=MLP_HIDDEN_LAYERS,
        dropout_rate=MLP_DROPOUT_RATE
    ).to(DEVICE)
    logging.info(f"\nMLP模型已初始化并移动到 {DEVICE}。\n{model}")

    # 训练模型
    history = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE, DEVICE)

    # 加载性能最佳的模型进行最终评估
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_mlp_model_no_loc.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    # 评估模型并绘图
    evaluate_model(model, test_loader, y_scaler, DEVICE, X_train_s.shape[1])
    plot_history(history)

    logging.info("\n--- 所有流程执行完毕 ---")


if __name__ == '__main__':
    main()
# ==============================================================================

