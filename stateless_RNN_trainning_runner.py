import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
import random
import argparse
from typing import List, Dict, Tuple, Any

# --- 全局设备设置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 1. 模型定义 (保持不变) ---
# ==============================================================================
class StatelessRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int, model_type: str = 'GRU'):
        super(StatelessRNN, self).__init__()
        if model_type.upper() not in ['GRU', 'LSTM']:
            raise ValueError("model_type must be 'GRU' or 'LSTM'")
        RNNCell = nn.GRU if model_type.upper() == 'GRU' else nn.LSTM
        self.rnn = RNNCell(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rnn_output, _ = self.rnn(x)
        return self.fc(rnn_output[:, -1, :])


# ==============================================================================

# --- 2. 数据加载与准备 ("一次性物化"策略) ---
# ==============================================================================
def create_sequences(trajectories: List[pd.DataFrame], feature_cols: List[str], target_cols: List[str],
                     seq_length: int) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """将轨迹列表转换为适用于Stateless RNN的滑动窗口序列。"""
    all_sequences_x, all_sequences_y = [], []
    for traj_df in trajectories:
        if len(traj_df) < seq_length + 1:
            continue
        x_data = traj_df[feature_cols].values
        y_data = traj_df[target_cols].values
        for i in range(len(traj_df) - seq_length):
            all_sequences_x.append(x_data[i:i + seq_length])
            all_sequences_y.append(y_data[i + seq_length])
    if not all_sequences_x:
        return None, None
    return np.array(all_sequences_x), np.array(all_sequences_y)


def load_and_prepare_data(filepath: str, seq_length: int, random_seed: int, model_output_dir: str) -> Tuple:
    logging.info(f"--- 步骤 1: 加载并准备数据 (序列长度: {seq_length}) ---")
    logging.warning("采用'一次性物化'策略，初始化内存消耗较大，可能需要较长时间...")
    with open(filepath, 'rb') as f:
        all_trajectories = pickle.load(f)

    feature_cols = ['era5_wind_speed', 'era5_wind_dir_sin', 'era5_wind_dir_cos', 'era5_swh', 'era5_mwp',
                    'era5_wave_dir_sin', 'era5_wave_dir_cos']
    target_cols = ['delta_u', 'delta_v']

    for traj in all_trajectories:
        traj['delta_u'] = traj['ve'] - traj['hycom_u']
        traj['delta_v'] = traj['vn'] - traj['hycom_v']

    train_val_traj, test_traj = train_test_split(all_trajectories, test_size=0.15, random_state=random_seed)
    train_traj, val_traj = train_test_split(train_val_traj, test_size=0.1, random_state=random_seed)

    logging.info("正在为所有数据集创建滑动窗口序列...")
    X_train, y_train = create_sequences(train_traj, feature_cols, target_cols, seq_length)
    X_val, y_val = create_sequences(val_traj, feature_cols, target_cols, seq_length)
    X_test, y_test = create_sequences(test_traj, feature_cols, target_cols, seq_length)
    if X_train is None: return (None,) * 8

    logging.info("正在标准化数据...")
    num_features = X_train.shape[2]
    x_scaler = StandardScaler().fit(X_train.reshape(-1, num_features))
    y_scaler = StandardScaler().fit(y_train)

    X_train_s = x_scaler.transform(X_train.reshape(-1, num_features)).reshape(X_train.shape)
    X_val_s = x_scaler.transform(X_val.reshape(-1, num_features)).reshape(X_val.shape)
    X_test_s = x_scaler.transform(X_test.reshape(-1, num_features)).reshape(X_test.shape)
    y_train_s = y_scaler.transform(y_train)
    y_val_s = y_scaler.transform(y_val)

    os.makedirs(model_output_dir, exist_ok=True)
    with open(os.path.join(model_output_dir, f'x_scaler_seq{seq_length}.pkl'), 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(os.path.join(model_output_dir, f'y_scaler_seq{seq_length}.pkl'), 'wb') as f:
        pickle.dump(y_scaler, f)

    logging.info("数据准备完成。")
    return X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test, x_scaler, y_scaler, val_traj


# ==============================================================================

# --- 3. 训练与评估工具 (与之前脚本类似) ---
# ==============================================================================
# `train_model`, `evaluate_model`, `plot_history`, `plot_trajectory_prediction`
# 将在这里定义 (为简洁起见，此处省略，但它们在完整脚本中)
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

def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """计算 MAPE，注意：当真实值接近0时，该指标可能不稳定。"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    # 避免除以零
    denominator = np.maximum(np.abs(y_true), epsilon)
    return np.mean(np.abs((y_true - y_pred) / denominator)) * 100

def train_model(model: nn.Module, loaders: Dict, scalers: Dict, val_trajectories: List[pd.DataFrame], args: argparse.Namespace, exp_dir: str):
    logging.info("\n--- 步骤 3: 开始模型训练 ---")
    criterion = nn.MSELoss()
    # 新增 L2 正则化（weight decay）
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=16)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.lr_patience)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_model_path = os.path.join(exp_dir, 'best_model.pth')

    vis_traj = random.choice([t for t in val_trajectories if len(t) > args.sequence_length + 5])

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0.0
        for X_batch, y_batch in loaders['train']:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        history['train_loss'].append(total_train_loss / len(loaders['train']))

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for X_batch, y_batch in loaders['val']:
                outputs = model(X_batch.to(DEVICE))
                val_preds.append(outputs.cpu().numpy())
                val_targets.append(y_batch.numpy())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        avg_val_loss = np.mean((val_preds - val_targets) ** 2)
        val_r2 = r2_score(val_targets, val_preds)
        history['val_loss'].append(avg_val_loss)
        history['val_r2'].append(val_r2)

        # learning rate 调度器
        scheduler.step(avg_val_loss)

        logging.info(f"Epoch [{epoch + 1:03d}/{args.epochs:03d}] | 训练损失: {history['train_loss'][-1]:.6f} | 验证损失: {avg_val_loss:.6f} | 验证 R²: {val_r2:.4f}")

        if (epoch + 1) % 5 == 0:
            plot_trajectory_prediction(model, vis_traj, scalers['x'], scalers['y'], args.sequence_length, epoch + 1, exp_dir)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"  -> 验证损失改善，已保存最佳模型到 '{best_model_path}'")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logging.info(f"连续 {args.patience} 个 epoch 验证损失未改善，触发早停。")
                break
    return history

def adjusted_r2_score(r2: float, n: int, p: int) -> float:
    return 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) != 0 else r2

def evaluate_model(model: nn.Module, test_loader: DataLoader, y_scaler: StandardScaler, num_features: int):
    logging.info("\n--- 步骤 4: 在测试集上评估最终模型 ---")
    model.eval()
    preds_scaled, targets_original_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch.to(DEVICE))
            preds_scaled.append(outputs.cpu().numpy())
            targets_original_list.append(y_batch.numpy())

    preds_scaled = np.concatenate(preds_scaled)
    targets_original = np.concatenate(targets_original_list)
    preds_original = y_scaler.inverse_transform(preds_scaled)

    r2_u = r2_score(targets_original[:, 0], preds_original[:, 0])
    r2_v = r2_score(targets_original[:, 1], preds_original[:, 1])
    r2_overall = r2_score(targets_original, preds_original)
    adj_r2_overall = adjusted_r2_score(r2_overall, len(targets_original), num_features)
    mae = np.mean(np.abs(preds_original - targets_original))
    rmse = np.sqrt(np.mean((preds_original - targets_original) ** 2))

    logging.info("最终模型在测试集上的性能:")
    logging.info(f"  - MAE (m/s): {mae:.4f}, RMSE (m/s): {rmse:.4f}")
    logging.info(f"  - R² (U): {r2_u:.4f}, R² (V): {r2_v:.4f}")
    logging.info(f"  - R² (总体): {r2_overall:.4f}, Adjusted R² (总体): {adj_r2_overall:.4f}")


def plot_history(history: Dict[str, List[float]], exp_dir: str):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(history['train_loss'], label='训练损失', color='tab:blue')
    ax1.plot(history['val_loss'], label='验证损失', color='tab:orange')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue'); ax1.legend(loc='upper left'); ax1.grid(True)
    ax2 = ax1.twinx()
    ax2.plot(history['val_r2'], label='验证 $R^2$', color='tab:green', linestyle='--')
    ax2.set_ylabel('$R^2$', color='tab:green'); ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')
    ax1.set_title('RNN模型训练过程', fontsize=14, pad=20)
    fig.tight_layout(pad=2.0)
    plot_path = os.path.join(exp_dir, 'loss_r2_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"训练历史曲线图已保存到: '{plot_path}'")

def plot_trajectory_prediction(model: nn.Module, trajectory: pd.DataFrame, x_scaler: StandardScaler,
                               y_scaler: StandardScaler, seq_length: int, epoch: int, exp_dir: str):
    model.eval()
    feature_cols = ['era5_wind_speed', 'era5_wind_dir_sin', 'era5_wind_dir_cos', 'era5_swh', 'era5_mwp', 'era5_wave_dir_sin', 'era5_wave_dir_cos']
    target_cols = ['delta_u', 'delta_v']

    x_traj_seq, y_traj_true = create_sequences([trajectory], feature_cols, target_cols, seq_length)
    if x_traj_seq is None: return

    x_traj_scaled = x_scaler.transform(x_traj_seq.reshape(-1, x_traj_seq.shape[2])).reshape(x_traj_seq.shape)
    x_tensor = torch.FloatTensor(x_traj_scaled).to(DEVICE)

    with torch.no_grad():
        y_pred_scaled = model(x_tensor).cpu().numpy()
    y_pred_true = y_scaler.inverse_transform(y_pred_scaled)
    time_axis = trajectory.iloc[seq_length: seq_length + len(y_traj_true)]['time']

    plt.rcParams['font.sans-serif'] = ['SimHei', 'SimSun', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 7))
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, y_traj_true[:, 0], label='真实值', color='blue', alpha=0.7)
    plt.plot(time_axis, y_pred_true[:, 0], label='预测值', color='red', linestyle='--')
    plt.title(f'Epoch {epoch}: U-分量残差预测 vs. 真实')
    plt.ylabel('$\\Delta U$ (m/s)'); plt.legend(); plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time_axis, y_traj_true[:, 1], label='真实值', color='blue', alpha=0.7)
    plt.plot(time_axis, y_pred_true[:, 1], label='预测值', color='red', linestyle='--')
    plt.title(f'Epoch {epoch}: V-分量残差预测 vs. 真实')
    plt.xlabel('时间'); plt.ylabel('$\\Delta V$ (m/s)'); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(exp_dir, f'prediction_epoch_{epoch:03d}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"  -> 可视化预测图已保存到: '{plot_path}'")


# ==============================================================================

# --- 4. 主执行流程 ---
# ==============================================================================
def main(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{timestamp}_seq{args.sequence_length}_{args.model_type.lower()}_hid{args.hidden_size}_lay{args.num_layers}"
    exp_dir = os.path.join(args.results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    log_path = os.path.join(exp_dir, 'training.log')
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()])

    logging.info(f"--- 启动新的Stateless RNN实验 (Eager Materialization模式) ---")
    logging.info(f"实验目录: {exp_dir}");
    [logging.info(f"  - {key}: {value}") for key, value in vars(args).items()]

    data = load_and_prepare_data(args.data_path, args.sequence_length, args.random_seed, exp_dir)
    (X_train_s, y_train_s, X_val_s, y_val_s, X_test_s, y_test,
     x_scaler, y_scaler, val_traj) = data

    if X_train_s is None:
        logging.error("数据准备失败，终止实验。");
        return

    loaders = {
        'train': DataLoader(TensorDataset(torch.from_numpy(X_train_s).float(), torch.from_numpy(y_train_s).float()),
                            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(TensorDataset(torch.from_numpy(X_val_s).float(), torch.from_numpy(y_val_s).float()),
                          batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(TensorDataset(torch.from_numpy(X_test_s).float(), torch.from_numpy(y_test).float()),
                           batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    }
    scalers = {'x': x_scaler, 'y': y_scaler}
    num_features = X_train_s.shape[2]

    model = StatelessRNN(
        input_size=num_features, hidden_size=args.hidden_size, num_layers=args.num_layers,
        output_size=y_train_s.shape[1], model_type=args.model_type
    ).to(DEVICE)
    logging.info(f"\n模型已初始化:\n{model}")

    history = train_model(model, loaders, scalers, val_traj, args, exp_dir)

    logging.info("\n加载性能最佳的模型进行最终评估...")
    best_model_path = os.path.join(exp_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    evaluate_model(model, loaders['test'], y_scaler, num_features)
    plot_history(history, exp_dir)

    logging.info(f"\n--- 实验 {exp_name} 执行完毕 ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Stateless RNN 实验运行器 (Eager Materialization版)")
    parser.add_argument('--data_path', type=str, default='./processed_data/trajectories_sanitized.pkl',
                        help='清洗后的数据文件路径')
    parser.add_argument('--results_dir', type=str, default='./results/stateless_rnn', help='保存所有实验结果的根目录')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    parser.add_argument('--sequence_length', type=int, default=24, help='输入序列的历史窗口长度')
    # 模型架构参数
    parser.add_argument('--model_type', type=str, default='GRU', choices=['GRU', 'LSTM'], help='RNN单元类型')
    parser.add_argument('--hidden_size', type=int, default=64, help='RNN隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=2, help='RNN层数')
    # 训练超参数
    parser.add_argument('--learning_rate', type=float, default=0.001, help='优化器学习率')
    parser.add_argument('--batch_size', type=int, default=512, help='训练批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='最大训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心值')
    # 正则化参数（新增）
    parser.add_argument('--weight_decay', type=float, default=0.0, help='L2正则化(权重衰减)系数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')
    # learning rate patience 参数（新增）
    parser.add_argument('--lr_patience', type=int, default=5, help='学习率调度器的耐心值')

    args = parser.parse_args()
    main(args)