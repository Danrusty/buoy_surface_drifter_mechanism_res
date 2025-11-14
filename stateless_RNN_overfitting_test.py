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
import random
from typing import List, Dict, Tuple, Any

# --- 1. 配置与设置 ---
# ==============================================================================
LOG_DIR = './logs'
MODEL_OUTPUT_DIR = './models'
RESULTS_OUTPUT_DIR = './results/stateless_rnn_overfit_test' # 使用新的结果目录
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

log_filename = f"overfitting_test_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# --- 全局常量与超参数 ---
FINAL_DATA_PATH = './processed_data/trajectories_sanitized.pkl' # 假设这是您清洗后的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 诊断测试专用超参数 ---
# [MODIFIED] 选取极小的数据子集
N_TRAIN_TRAJ = 300
N_VAL_TRAJ = 50
N_TEST_TRAJ = 20
# [MODIFIED] 给予充分的时间过拟合
EPOCHS = 300
EARLY_STOPPING_PATIENCE = 300 # 禁用早停

# 原始模型参数
SEQUENCE_LENGTH = 6
LEARNING_RATE = 0.0001
BATCH_SIZE = 512
RANDOM_SEED = 42
HIDDEN_SIZE = 64
NUM_LAYERS = 3
VISUALIZATION_EPOCH_INTERVAL = 5

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
# ==============================================================================


# --- 2. 模型定义 (保持不变) ---
# ==============================================================================
class StatelessGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(StatelessGRU, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_out, _ = self.rnn(x)
        last_time_step_out = rnn_out[:, -1, :]
        return self.fc(last_time_step_out)
# ==============================================================================


# --- 3. 数据加载与准备 (已修改) ---
# ==============================================================================
def create_sequences(trajectories: List[pd.DataFrame], feature_cols: List[str], target_cols: List[str],
                     seq_length: int) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """将轨迹列表转换为适用于Stateless RNN的滑动窗口序列。"""
    logging.info(f"--- 正在创建长度为 {seq_length} 的时间序列窗口... ---")
    all_sequences_x, all_sequences_y = [], []
    for traj_df in trajectories:
        if len(traj_df) < seq_length + 1: continue
        x_data = traj_df[feature_cols].values
        y_data = traj_df[target_cols].values
        for i in range(len(traj_df) - seq_length):
            all_sequences_x.append(x_data[i:i + seq_length])
            all_sequences_y.append(y_data[i + seq_length])
    if not all_sequences_x:
        logging.warning("没有创建任何序列，可能是轨迹太短或数据有问题。")
        return None, None
    return np.array(all_sequences_x), np.array(all_sequences_y)


def load_and_prepare_data(filepath: str, seq_length: int) -> Tuple:
    """[已修改] 加载、清洗、并使用极小的数据子集进行划分、创建序列和标准化。"""
    logging.info(f"--- 步骤 1: 从 '{filepath}' 加载数据 ---")
    try:
        with open(filepath, 'rb') as f:
            all_trajectories = pickle.load(f)
    except FileNotFoundError:
        logging.error(f"错误: 数据文件未找到！")
        return (None,) * 9

    logging.info(f"原始数据集共包含 {len(all_trajectories)} 条轨迹。")

    # [MODIFIED] 关键修改：执行过拟合测试的数据子集化
    logging.warning("--- 正在执行诊断：过拟合测试 ---")
    logging.warning(f"将仅使用 {N_TRAIN_TRAJ} (训练) + {N_VAL_TRAJ} (验证) + {N_TEST_TRAJ} (测试) 条轨迹。")

    # 清洗数据（假设您已在预处理中清除了NaN渗漏，这里再做一个保险）
    feature_cols = [
        'era5_wind_speed', 'era5_wind_dir_sin', 'era5_wind_dir_cos',
        'era5_swh', 'era5_mwp', 'era5_wave_dir_sin', 'era5_wave_dir_cos',
        've', 'vn', 'hycom_u', 'hycom_v' # 确保原始值也存在
    ]
    cleaned_trajectories = []
    for i, traj in enumerate(all_trajectories):
        traj['delta_u'] = traj['ve'] - traj['hycom_u']
        traj['delta_v'] = traj['vn'] - traj['hycom_v']
        traj['trajectory_id'] = i

        # 移除任何包含NaN的关键行
        cleaned_traj = traj.dropna(subset=feature_cols + ['delta_u', 'delta_v'])
        if not cleaned_traj.empty and len(cleaned_traj) > seq_length + 5: # 确保轨迹足够长
            cleaned_trajectories.append(cleaned_traj)

    logging.info(f"数据清洗后，剩余 {len(cleaned_trajectories)} 条有效长轨迹。")

    # 检查是否有足够的轨迹用于测试
    total_needed = N_TRAIN_TRAJ + N_VAL_TRAJ + N_TEST_TRAJ
    if len(cleaned_trajectories) < total_needed:
        logging.error(f"错误：清洗后的轨迹数量 ({len(cleaned_trajectories)}) 不足 {total_needed} 条，无法执行子集测试。")
        return (None,) * 9

    # 随机打乱并切片
    random.shuffle(cleaned_trajectories)
    train_traj = cleaned_trajectories[:N_TRAIN_TRAJ]
    val_traj = cleaned_trajectories[N_TRAIN_TRAJ : N_TRAIN_TRAJ + N_VAL_TRAJ]
    test_traj = cleaned_trajectories[N_TRAIN_TRAJ + N_VAL_TRAJ : total_needed]
    # [MODIFIED] 结束

    target_cols = ['delta_u', 'delta_v']
    feature_cols.remove('ve'); feature_cols.remove('vn'); feature_cols.remove('hycom_u'); feature_cols.remove('hycom_v')

    logging.info(f"子集划分完成: {len(train_traj)} (训练), {len(val_traj)} (验证), {len(test_traj)} (测试) 条轨迹")

    X_train, y_train = create_sequences(train_traj, feature_cols, target_cols, seq_length)
    X_val, y_val = create_sequences(val_traj, feature_cols, target_cols, seq_length)
    X_test, y_test = create_sequences(test_traj, feature_cols, target_cols, seq_length)

    if X_train is None or X_val is None or X_test is None:
        return (None,) * 9

    logging.info(f"滑动窗口创建完成: {len(X_train)} (训练), {len(X_val)} (验证), {len(X_test)} (测试) 个样本")

    num_features = X_train.shape[2]
    X_train_reshaped = X_train.reshape(-1, num_features)

    x_scaler = StandardScaler().fit(X_train_reshaped)
    y_scaler = StandardScaler().fit(y_train)

    X_train_scaled = x_scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = x_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_scaled = x_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)

    y_train_scaled = y_scaler.transform(y_train)
    y_val_scaled = y_scaler.transform(y_val)

    logging.info("数据标准化完成。")
    # ... (保存scaler的代码保持不变)
    with open(os.path.join(MODEL_OUTPUT_DIR, 'rnn_x_scaler_overfit.pkl'), 'wb') as f: pickle.dump(x_scaler, f)
    with open(os.path.join(MODEL_OUTPUT_DIR, 'rnn_y_scaler_overfit.pkl'), 'wb') as f: pickle.dump(y_scaler, f)

    return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, X_test_scaled, y_test, x_scaler, y_scaler, val_traj
# ==============================================================================


# --- 4. 训练与评估工具 (已修复绘图) ---
# ==============================================================================
def _run_inference_pass(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            outputs = model(X_batch.to(device))
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                val_trajectories: List[pd.DataFrame],
                x_scaler: StandardScaler, y_scaler: StandardScaler, device: torch.device):
    """[已修改] 训练模型，早停已通过全局变量禁用"""
    logging.info("\n--- 步骤 4: 开始模型训练 (过拟合测试模式) ---")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_stateless_rnn_model_overfit.pth')

    vis_traj = random.choice([t for t in val_trajectories if len(t) > SEQUENCE_LENGTH + 5]) if val_trajectories else None

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        history['train_loss'].append(total_train_loss / len(train_loader))

        # (验证逻辑保持不变)
        val_preds, val_targets = _run_inference_pass(model, val_loader, device)
        avg_val_loss = np.mean((val_preds - val_targets)**2)
        val_r2 = r2_score(val_targets, val_preds)
        history['val_loss'].append(avg_val_loss)
        history['val_r2'].append(val_r2)

        logging.info(
            f"Epoch [{epoch + 1:03d}/{EPOCHS:03d}] | 训练损失: {history['train_loss'][-1]:.6f} | 验证损失: {avg_val_loss:.6f} | 验证 R²: {val_r2:.4f}")

        if vis_traj is not None and (epoch + 1) % VISUALIZATION_EPOCH_INTERVAL == 0:
            plot_trajectory_prediction(model, vis_traj, x_scaler, y_scaler, SEQUENCE_LENGTH, epoch + 1, device)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  -> 验证损失改善，已保存最佳模型到 '{best_model_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE: # 这里的Patience被设为300
                logging.info(f"已达到 {EPOCHS} 轮训练，测试结束。")
                break
    return history

def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) != 0 else r2

def evaluate_model(model: nn.Module, test_loader: DataLoader, y_scaler: StandardScaler,
                   device: torch.device, num_features: int) -> None:
    """在测试集上评估最终模型"""
    logging.info("\n--- 步骤 5: 在测试集上评估最终模型 ---")
    preds_scaled, targets_original_np = _run_inference_pass(model, test_loader, device)
    preds_original = y_scaler.inverse_transform(preds_scaled)

    # (计算指标的代码保持不变)
    r2_u = r2_score(targets_original_np[:, 0], preds_original[:, 0])
    r2_v = r2_score(targets_original_np[:, 1], preds_original[:, 1])
    r2_overall = r2_score(targets_original_np, preds_original)
    adj_r2_overall = adjusted_r2_score(r2_overall, len(targets_original_np), num_features)
    mae = np.mean(np.abs(preds_original - targets_original_np))
    rmse = np.sqrt(np.mean((preds_original - targets_original_np) ** 2))

    logging.info("最终模型在测试集上的性能 (基于过拟合测试):")
    logging.info(f"  - R² (总体): {r2_overall:.4f}")
    logging.info(f"  - MAE (m/s): {mae:.4f}, RMSE (m/s): {rmse:.4f}")


def plot_history(history: Dict[str, List[float]]) -> None:
    """绘制训练过程中的损失和 R² 曲线图。"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(history['train_loss'], label='训练损失 (Training Loss)', color='tab:blue')
    ax1.plot(history['val_loss'], label='验证损失 (Validation Loss)', color='tab:orange')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('均方误差损失 (MSE Loss)', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.legend(loc='upper left'); ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(history['val_r2'], label='验证 $R^2$ (Validation $R^2$)', color='tab:green', linestyle='--')
    ax2.set_ylabel('决定系数 $R^2$', color='tab:green'); ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')
    ax1.set_title('RNN 过拟合测试', fontsize=14, pad=20)
    fig.tight_layout(pad=2.0)
    plot_path = os.path.join(RESULTS_OUTPUT_DIR, 'overfit_test_loss_r2_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"\n训练历史曲线图已保存到: '{plot_path}'")
    plt.close()


def plot_trajectory_prediction(model: nn.Module, trajectory: pd.DataFrame, x_scaler: StandardScaler,
                               y_scaler: StandardScaler, seq_length: int, epoch: int, device: torch.device):
    """[已修复] 在单条轨迹上进行预测并可视化结果。"""
    model.eval()
    feature_cols = [
        'era5_wind_speed', 'era5_wind_dir_sin', 'era5_wind_dir_cos',
        'era5_swh', 'era5_mwp', 'era5_wave_dir_sin', 'era5_wave_dir_cos'
    ]
    target_cols = ['delta_u', 'delta_v']
    x_traj_seq, y_traj_true = create_sequences([trajectory], feature_cols, target_cols, seq_length)
    if x_traj_seq is None: return

    x_traj_scaled = x_scaler.transform(x_traj_seq.reshape(-1, x_traj_seq.shape[2])).reshape(x_traj_seq.shape)
    x_tensor = torch.FloatTensor(x_traj_scaled).to(device)
    with torch.no_grad():
        y_pred_scaled = model(x_tensor).cpu().numpy()

    # [注意] 这里y_pred_true是错的，应该是y_pred_original
    y_pred_original = y_scaler.inverse_transform(y_pred_scaled)

    # [FIXED] 修复了时间轴切片错误
    time_axis = trajectory.iloc[seq_length: seq_length + len(y_traj_true)]['time']

    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, y_traj_true[:, 0], label='真实值', color='blue', alpha=0.7)
    plt.plot(time_axis, y_pred_original[:, 0], label='预测值', color='red', linestyle='--')
    plt.title(f'Epoch {epoch}: U-分量残差预测 vs. 真实 (轨迹ID: {trajectory["trajectory_id"].iloc[0]})')
    plt.ylabel('Delta U (m/s)'); plt.legend(); plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, y_traj_true[:, 1], label='真实值', color='blue', alpha=0.7)
    plt.plot(time_axis, y_pred_original[:, 1], label='预测值', color='red', linestyle='--')
    plt.title(f'Epoch {epoch}: V-分量残差预测 vs. 真实 (轨迹ID: {trajectory["trajectory_id"].iloc[0]})')
    plt.xlabel('时间'); plt.ylabel('Delta V (m/s)'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_OUTPUT_DIR, f'prediction_epoch_{epoch:03d}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(
        f"  -> 可视化预测图已保存到: '{plot_path}' (轨迹ID: {trajectory['trajectory_id'].iloc[0]}, 长度: {len(trajectory)})")
# ==============================================================================


# --- 5. 主执行流程 ---
# ==============================================================================
def main():
    logging.info(f"--- 启动Stateless RNN [过拟合测试]，将使用设备: {DEVICE} ---")

    data = load_and_prepare_data(FINAL_DATA_PATH, SEQUENCE_LENGTH)
    if data[0] is None:
        logging.error("数据准备失败，终止训练。")
        return
    X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test, x_scaler, y_scaler, val_traj = data

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train_s).float()),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_s).float(), torch.from_numpy(y_val_s).float()),
                            batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_s).float(), torch.from_numpy(y_test).float()),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = StatelessGRU(
        input_size=X_train_s.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=y_train_s.shape[1]
    ).to(DEVICE)
    logging.info(f"\nStateless GRU模型已初始化并移动到 {DEVICE}。\n{model}")

    history = train_model(model, train_loader, val_loader, val_traj, x_scaler, y_scaler, DEVICE)

    logging.info("\n加载性能最佳的模型进行最终评估...")
    best_model_path = os.path.join(MODEL_OUTPUT_DIR, 'best_stateless_rnn_model_overfit.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    evaluate_model(model, test_loader, y_scaler, X_train_s.shape[2], DEVICE)
    plot_history(history)

    logging.info("\n--- 所有流程执行完毕 ---")


if __name__ == '__main__':
    main()
# ==============================================================================
