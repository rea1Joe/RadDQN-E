import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 用户配置区域 ---
# ==============================================================================

# --- 【必须修改】请在此处填入您 RadDQN-E 算法5个种子实验生成的 actions 文件路径 ---
RAD_DQN_E_ACTION_FILES = [
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s1231_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s3007_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s6027_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        
        '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s9881_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',

]

# --- 【必须修改】您希望保存图表的文件夹 ---
SAVE_DIR = '/home/joe/RadDQN-main/Figures_RadDQN_E/4_Discussion/5paths'

# --- 地图配置 (坐标格式: X, Y) ---
# !! 注意：请确保这里的起点、终点和辐射源配置与您所分析的任务一致 !!
GRID_SIZE = 10
START_POS = (0, 6)
GOAL_POS = (6, 0)
SOURCES_POS = {'S1': (2, 0), 'S2': (4, 3), 'S3': (5, 8)}

# ==============================================================================
# --- 2. 脚本主逻辑 (已修改为生成单一热力图) ---
# ==============================================================================

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 计算RadDQN-E的访问频次矩阵 ---
visit_counts = np.zeros((GRID_SIZE, GRID_SIZE))
paths_loaded = 0

print("--- 正在处理 RadDQN-E 的路径数据 ---")
for path_file in RAD_DQN_E_ACTION_FILES:
    try:
        with open(path_file, 'rb') as f:
            paths = pickle.load(f)
        if paths:
            final_path = paths[-1]
            paths_loaded += 1
            # 累加每个坐标点的访问次数
            for (y, x) in final_path:
                if 0 <= y < GRID_SIZE and 0 <= x < GRID_SIZE:
                    visit_counts[y, x] += 1
    except FileNotFoundError:
        print(f"警告：找不到文件 {path_file}")
        continue
    except Exception as e:
        print(f"处理文件 {path_file} 时出错: {e}")

# --- 绘制热力图 ---
if paths_loaded > 0:
    # 创建一个独立的图和坐标轴
    fig, ax = plt.subplots(figsize=(12, 12))

    # 绘制热力图
    im = ax.imshow(visit_counts, cmap='hot', interpolation='nearest')
    
    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Visit Count in {paths_loaded} Final Paths')

    # 绘制静态地图元素
    ax.scatter(START_POS[0], START_POS[1], s=200, c='cyan', marker='^', label='Start', zorder=10, edgecolors='black')
    ax.scatter(GOAL_POS[0], GOAL_POS[1], s=200, c='magenta', marker='*', label='Goal', zorder=10, edgecolors='black')
    source_coords = np.array(list(SOURCES_POS.values()))
    if source_coords.size > 0:
        ax.scatter(source_coords[:, 0], source_coords[:, 1], s=300, c='yellow', marker='P', label='Radiation Source', zorder=5, edgecolors='black')

    # 设置图表样式
    ax.set_title('RadDQN-E Policy Consistency Heatmap (from 5 Final Paths)', fontsize=16)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.grid(True, which='both', color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.set_aspect('equal', adjustable='box')
    ax.legend()

    # 保存图表
    save_path = os.path.join(SAVE_DIR, 'path_consistency_heatmap_RadDQN-E.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nRadDQN-E 策略一致性热力图已保存至: {save_path}")
    plt.close()
else:
    print("\n未能加载任何有效的路径数据，无法生成图表。")