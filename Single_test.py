import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 用户配置区域：请在此处修改您的文件路径和设置 ---
# ==============================================================================



# --- 【必须修改】对应的 Actions 和 Rewards 文件名 ---
# !! 注意：这个文件名非常长，请从您的结果文件夹中完整复制 !!
ACTIONS_FILENAME = '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s9881_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle' # <-- !! 修改这里 !!
REWARDS_FILENAME = '/home/joe/RadDQN-main/Logdir_RadDQN_E/8.24/Reward_down_w_energy_r0.5_e0.5_d_0.5_s9881_5000_Uncerntainty_S1_85_25_S2_34_5_S3_02_5/rewards_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle' # <-- !! 修改这里 !!

# --- 【必须修改】您希望保存图片的文件夹 ---
SAVE_DIR = '/home/joe/RadDQN-main/Figures_RadDQN_E/test/8.25/Discussion_short_sight' # <-- !! 修改这里 !!
# 
# --- 地图和智能体配置 (必须与训练时使用的配置一致) ---
GRID_SIZE = 10
START_POS = (0, 6)
GOAL_POS = (6, 0)
SOURCES_POS = {
    'S1': [8, 5],
    'S2': [3, 4],
    'S3': [0, 2]
}

# --- 平滑曲线的窗口大小 (值越大，曲线越平滑) ---
SMOOTHING_WINDOW = 100

# ==============================================================================
# --- 脚本主逻辑：通常无需修改以下内容 ---
# ==============================================================================

# 构造完整文件路径
actions_path =  ACTIONS_FILENAME
rewards_path =  REWARDS_FILENAME

# 自动创建保存目录
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"已创建保存目录: {SAVE_DIR}")

# --- 2. 绘制最终路径图 ---
print("--- 正在生成路径图 ---")
plt.figure(figsize=(10, 10))

try:
    with open(actions_path, 'rb') as f:
        paths = pickle.load(f)
    if paths:
        final_path = np.array(paths[-1])
        plt.plot(final_path[:, 1], final_path[:, 0], marker='o', color='blue', linestyle='-', label='Agent Path')
        print(f"路径加载成功，共 {len(paths)} 个回合，显示最后一个回合的路径。")
    else:
        print("警告：路径文件为空。")
except FileNotFoundError:
    print(f"错误：找不到路径文件 '{actions_path}'。")
except Exception as e:
    print(f"处理路径数据时出错: {e}")

# 设置图表样式
plt.title('Final Path of the Agent', fontsize=16)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)
plt.scatter(START_POS[1], START_POS[0], s=200, c='green', marker='^', label='Start', zorder=5)
plt.scatter(GOAL_POS[1], GOAL_POS[0], s=200, c='purple', marker='*', label='Goal', zorder=5)

source_coords = np.array(list(SOURCES_POS.values()))
if source_coords.size > 0:
    plt.scatter(source_coords[:, 1], source_coords[:, 0], s=300, c='orange', marker='P', label='Radiation Source', zorder=5, edgecolors='black')

plt.legend()
plt.grid(True)
plt.xlim(-0.5, GRID_SIZE - 0.5)
plt.ylim(-0.5, GRID_SIZE - 0.5)
plt.gca().invert_yaxis()
plt.xticks(np.arange(GRID_SIZE))
plt.yticks(np.arange(GRID_SIZE))
plt.gca().set_aspect('equal', adjustable='box')

path_save_path = os.path.join(SAVE_DIR, 'single_run_path.png')
plt.savefig(path_save_path)
print(f"路径图已保存至: {path_save_path}")
plt.close() # 关闭图形，为下一个图做准备

# --- 3. 绘制平滑奖励曲线 ---
print("\n--- 正在生成奖励曲线图 ---")

def moving_average(data, window_size):
    if len(data) < window_size:
        return np.array([])
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

plt.figure(figsize=(12, 6))

try:
    with open(rewards_path, 'rb') as f:
        rewards = pickle.load(f)
    
    # 绘制原始数据（设为半透明）
    plt.plot(rewards, color='deepskyblue', alpha=0.3, label='Raw Rewards')
    
    # 绘制平滑后的数据
    smoothed_rewards = moving_average(rewards, SMOOTHING_WINDOW)
    if smoothed_rewards.size > 0:
        plt.plot(smoothed_rewards, color='blue', linewidth=2, label=f'Smoothed Rewards (window={SMOOTHING_WINDOW})')
    
    print(f"奖励数据加载成功，共 {len(rewards)} 个回合。")
except FileNotFoundError:
    print(f"错误：找不到奖励文件 '{rewards_path}'。")
except Exception as e:
    print(f"处理奖励数据时出错: {e}")

plt.title('Reward per Episode', fontsize=16)
plt.xlabel('Episode', fontsize=12)
plt.ylabel('Cumulative Reward', fontsize=12)
plt.legend()
plt.grid(True)

reward_save_path = os.path.join(SAVE_DIR, 'single_run_reward_curve.png')
plt.savefig(reward_save_path)
print(f"奖励图已保存至: {reward_save_path}")
plt.close()

print("\n脚本执行完毕。")