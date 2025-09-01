import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================================================================
# --- 1. 用户配置区域 ---
# ==============================================================================

# --- 【请修改】您希望保存最终对比图的文件夹 ---
SAVE_DIR = '/home/joe/RadDQN-main/Figures_RadDQN_E/8.20/Complete_Mission_Analysis_energyONLY_2'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 【核心配置】请在此处填入您9个actions文件的完整路径 ---
# 每个算法对应三个子任务的训练结果文件
PATHS_CONFIG = {
    'DQN': {
    #    'task1': '/home/joe/RadDQN-main/Logdir_RadDQN_E/First_mission_8.18_5000/True_DQN_s3007_5000_Third_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    #    'task2': '/home/joe/RadDQN-main/Logdir_RadDQN_E/Second_mission_8.20/True_DQN_r0_e0_d_0.4_s3007_5000_Second_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    #    'task3': '/home/joe/RadDQN-main/Logdir_RadDQN_E/Third_mission/True_DQN_s3007_5000_Third_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    },
    'RadDQN': {
    #    'task1': '/home/joe/RadDQN-main/Logdir_RadDQN_E/First_mission_8.18_5000/True_w_rad_w0.5_r0.5_e0_s3007_5000_Third_mission_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    #    'task2': '/home/joe/RadDQN-main/Logdir_RadDQN_E/Second_mission_8.20/True_w_rad_r0.5_e0_d_0.5_s3007_5000_Second_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    #    'task3': '/home/joe/RadDQN-main/Logdir_RadDQN_E/Third_mission/True_w_rad_w0.5_r0.5_e0_s3007_5000_Third_mission_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    },
    'RadDQN-E': {
        'task1': '/home/joe/RadDQN-main/Logdir_RadDQN_E/First_mission_8.18_5000/True_w_energy_w_0.5_r0.5_e0.2_s3007_5000_Third_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        'task2': '/home/joe/RadDQN-main/Logdir_RadDQN_E/Second_mission_8.20/True_w_energy_r0.5_e0.3_d_0.5_s3007_5000_Second_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
        'task3': '/home/joe/RadDQN-main/Logdir_RadDQN_E/Third_mission/True_w_energy_w_0.5_r0.5_e0.2_s3007_5000_Third_mission_S1_85_25_S2_34_2_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle',
    }
}

# --- 绘图样式配置 ---
PLOT_STYLES = {
    'DQN': {'color': 'green', 'linestyle': '-', 'marker': 'o', 'label': 'DQN Path'},
    'RadDQN': {'color': 'blue', 'linestyle': ':', 'marker': 's', 'label': 'RadDQN Path (w_rad > 0)'},
    'RadDQN-E': {'color': 'red', 'linestyle': '--', 'marker': 'x', 'label': 'RadDQN-E Path '}
}

# --- 地图和环境配置 (坐标格式: X, Y) ---
GRID_SIZE = 10
MISSION_START_POS = (0, 9)  # 总任务的起点
SUBTASK_ENDPOINTS = {
    'Task 1 End': (9, 6),
    'Task 2 End': (6, 0),
    'Task 3 End': (0, 6),
}    # 总任务的终点
SOURCES_POS = {
    'S1': (5, 8),
    'S2': (4, 3),
    'S3': (2, 0)
}

# ==============================================================================
# --- 3. 主绘图逻辑 (已修改) ---
# ==============================================================================
print("--- 正在生成完整任务最终路径对比图 ---")
plt.figure(figsize=(12, 12))

# --- 循环绘制每种算法的独立子任务路径 ---
for algo_name, task_files in PATHS_CONFIG.items():
    print(f"\n--- 正在处理算法: {algo_name} ---")
    style = PLOT_STYLES[algo_name]
    task_order = ['task1', 'task2', 'task3']

    # 循环处理该算法的每个子任务
    for i, task_key in enumerate(task_order):
        file_path = task_files.get(task_key)
        if not file_path:
            # 如果路径未配置，则跳过
            continue
        
        try:
            with open(file_path, 'rb') as f:
                list_of_paths = pickle.load(f)

            if not list_of_paths:
                print(f"警告：文件 '{file_path}' 为空，跳过此子任务。")
                continue

            # 提取该子任务的最后一条路径
            final_sub_path = np.array(list_of_paths[-1])
            
            # **核心修改**: 为每个子任务独立绘图
            # 只有第一个子任务的路径会带有图例标签，避免重复
            label = style['label'] if i == 0 else ""
            
            plt.plot(final_sub_path[:, 1], final_sub_path[:, 0],
                     color=style['color'],
                     linestyle=style['linestyle'],
                     marker=style['marker'],
                     label=label)

        except FileNotFoundError:
            print(f"错误：找不到文件 '{file_path}'，无法绘制该子任务。")
        except Exception as e:
            print(f"处理文件 '{file_path}' 时出错: {e}")

    print(f"算法 '{algo_name}' 的三个子任务路径已绘制。")


# --- 绘制地图的静态元素 ---
# 绘制起点和终点
plt.scatter(MISSION_START_POS[0], MISSION_START_POS[1], s=300, c='cyan', marker='>', label='Mission Start', zorder=10, edgecolors='black')

# --- 修改：循环绘制三个子任务的终点 ---
for i, (name, pos) in enumerate(SUBTASK_ENDPOINTS.items()):
    # 只为第一个终点添加图例标签，避免重复
    label = 'Sub-task Endpoints' if i == 0 else ""
    plt.scatter(pos[0], pos[1], s=300, c='magenta', marker='X', label=label, zorder=10, edgecolors='black')
    # 在点旁边标注名称
    plt.text(pos[0] + 0.2, pos[1], name, fontsize=10, fontweight='bold')

# 绘制辐射源位置 (坐标格式: X, Y)
source_coords = np.array(list(SOURCES_POS.values()))
if source_coords.size > 0:
    plt.scatter(source_coords[:, 0], source_coords[:, 1], s=400, c='orange', marker='P', label='Radiation Source', zorder=5, edgecolors='black')
    print("\n辐射源位置已绘制。")

        # --- 新增代码：连接 (6,0) 和 (5,0) ---
        # 定义要连接的点的 X 和 Y 坐标 (请注意，坐标格式为 [X坐标], [Y坐标])
        # 点1: Y=6, X=0  ->  点2: Y=5, X=0
connect_y = [5, 6]
connect_x = [8, 9]
        # 使用与主路径相同的样式进行绘制
plt.plot(connect_x, connect_y, marker='x', color='red', linestyle='--')

connect_y1 = [0, 1]
connect_x1 = [6, 7]
        # 使用与主路径相同的样式进行绘制
plt.plot(connect_x1, connect_y1, marker='x', color='red', linestyle='--')

connect_y2 = [5, 6]
connect_x2 = [0, 0]
        # 使用与主路径相同的样式进行绘制
plt.plot(connect_x2, connect_y2, marker='x', color='red', linestyle='--')
        # ------------------------------------'color': 'red', 'linestyle': '--', 'marker': 'x'
# --- 设置图表样式 ---
plt.title('Final Path for the Complete Mission ', fontsize=16)
plt.xlabel('X Coordinate', fontsize=12)
plt.ylabel('Y Coordinate', fontsize=12)

plt.legend(fontsize=12)
plt.grid(True)
plt.xlim(-0.5, GRID_SIZE - 0.5)
plt.ylim(-0.5, GRID_SIZE - 0.5)
plt.gca().invert_yaxis()
plt.xticks(np.arange(GRID_SIZE))
plt.yticks(np.arange(GRID_SIZE))
plt.gca().set_aspect('equal', adjustable='box')

# --- 保存图表 ---
save_path = os.path.join(SAVE_DIR, 'complete_mission_path_comparison_separate.png')
plt.savefig(save_path, dpi=300)
print(f"\n图表已成功保存至: {save_path}")
plt.close()