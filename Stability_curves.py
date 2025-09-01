import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 用户配置区域 ---
# ==============================================================================

# --- 【必须修改】请在此处填入您5个种子实验生成的 rewards 文件路径 ---
REWARD_FILES = {
    'DQN': [
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/DQN_r0.5_e0.5_d_0.5_s1231_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/DQN_r0.5_e0.5_d_0.5_s3007_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/DQN_r0.5_e0.5_d_0.5_s6027_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/DQN_r0.5_e0.5_d_0.5_s7384_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/DQN_r0.5_e0.5_d_0.5_s9881_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    ],
    'RadDQN': [
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/rad_r0.5_e0.5_d_0.5_s1231_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/rad_r0.5_e0.5_d_0.5_s3007_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/rad_r0.5_e0.5_d_0.5_s6027_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/rad_r0.5_e0.5_d_0.5_s7384_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/rad_r0.5_e0.5_d_0.5_s9881_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        
    ],
    'RadDQN-E': [
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s1231_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s3007_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s6027_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s7384_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s9881_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        # ... etc for all 5 seeds
    ]
}

# --- 【必须修改】您希望保存图表的文件夹 ---
SAVE_DIR = '/home/joe/RadDQN-main/Figures_RadDQN_E/Uncertainty_Analysis'

# --- 平滑曲线的窗口大小 ---
SMOOTHING_WINDOW = 100

# ==============================================================================
# --- 2. 脚本主逻辑 ---
# ==============================================================================

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 创建一个1x3的子图布局
fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
fig.suptitle('Algorithm Performance Stability Across 5 Random Seeds', fontsize=20)

plot_colors = {'DQN': 'green', 'RadDQN': 'blue', 'RadDQN-E': 'red'}
algo_names = ['DQN', 'RadDQN', 'RadDQN-E']

for i, algo_name in enumerate(algo_names):
    ax = axes[i]
    reward_paths = REWARD_FILES.get(algo_name, [])
    
    if not reward_paths or None in reward_paths:
        print(f"警告：算法 '{algo_name}' 的文件路径未完全配置，跳过。")
        continue

    all_rewards_smoothed = []
    min_len = float('inf')

    # 加载并平滑所有数据
    for path in reward_paths:
        try:
            with open(path, 'rb') as f:
                rewards = pickle.load(f)
            smoothed = moving_average(rewards, SMOOTHING_WINDOW)
            all_rewards_smoothed.append(smoothed)
            if len(smoothed) < min_len:
                min_len = len(smoothed)
        except FileNotFoundError:
            print(f"错误：找不到文件 {path}")
            continue
    
    if not all_rewards_smoothed:
        continue

    # 将所有平滑后的曲线裁剪到最短长度，以便计算统计量
    rewards_matrix = np.array([arr[:min_len] for arr in all_rewards_smoothed])

    # 计算平均值和标准差
    mean_rewards = np.mean(rewards_matrix, axis=0)
    std_rewards = np.std(rewards_matrix, axis=0)
    
    x_axis = np.arange(len(mean_rewards))

    # 绘制平均值曲线
    ax.plot(x_axis, mean_rewards, color=plot_colors[algo_name], label=f'Mean Rewards ({algo_name})')
    # 填充标准差区域
    ax.fill_between(x_axis, mean_rewards - std_rewards, mean_rewards + std_rewards, color=plot_colors[algo_name], alpha=0.2, label=f'Std. Dev. ({algo_name})')
    
    ax.set_title(f'Stability of {algo_name}', fontsize=16)
    ax.set_xlabel('Episode', fontsize=12)
    ax.grid(True)
    ax.legend()

axes[0].set_ylabel('Cumulative Reward (Moving Average)', fontsize=12)

# 保存图表
save_path = os.path.join(SAVE_DIR, 'reward_stability_comparison.png')
plt.savefig(save_path, dpi=300)
print(f"\n性能稳定性图已保存至: {save_path}")
plt.close()