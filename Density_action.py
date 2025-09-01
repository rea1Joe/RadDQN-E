import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import Axes3D # 导入三维绘图工具
import os

# ==============================================================================
# --- 1. 请在这里配置您的文件路径和地图信息 ---
# ==============================================================================

# --- 【新增/修改】在这里指定您希望保存图表的文件夹 ---
SAVE_DIR = '/home/joe/RadDQN-main/Figures_RadDQN_E/8.24/reli_Rad_third_mission_s3007' # <-- !! 修改这里为您想要的路径 !!

# --- 结果文件的完整路径 ---
ACTIONS_PICKLE_PATH = '/home/joe/RadDQN-main/Logdir_RadDQN_E/Third_mission/True_w_rad_w0.5_r0.5_e0_s3007_5000_Third_mission_S1_85_25_S2_34_5_S3_02_5/actions_per_game_static_5000_epochs_2000_memsize_30_batch_size_600_sync_freq_1e-3_lr_leaky_func__S1_25_8_5_S2_5_3_4_S3_5_0_2_sync_improv_True_vanilla_exploration_False_partially_blind_False_.pickle'

# --- 地图配置 ---
GRID_SIZE = 10

# --- 【新增配置】设置3D柱状图的基座高度 ---
# 值为0时，柱子底部贴着热力图。增加该值可将柱子向上平移。
BAR_BASE_HEIGHT = 3000

# --- 自动创建保存目录 (如果不存在) ---
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
    print(f"已创建保存目录: {SAVE_DIR}")

# ==============================================================================
# --- 2. 加载并处理路径数据 (这部分代码不变) ---
# ==============================================================================
print(f"--- 正在加载并处理路径数据: {ACTIONS_PICKLE_PATH} ---")
try:
    with open(ACTIONS_PICKLE_PATH, 'rb') as f:
        list_of_paths = pickle.load(f)
    visit_counts = np.zeros((GRID_SIZE, GRID_SIZE))
    for path in list_of_paths:
        for (y, x) in path:
            if 0 <= y < GRID_SIZE and 0 <= x < GRID_SIZE:
                visit_counts[y, x] += 1
    print("路径数据处理完成。")
    data_loaded = True
except FileNotFoundError:
    print(f"错误：找不到文件 '{ACTIONS_PICKLE_PATH}'。")
    data_loaded = False
except Exception as e:
    print(f"处理数据时出错: {e}")
    data_loaded = False


# ==============================================================================
# --- 3. 绘制三维访问密度图 ---
# ==============================================================================
if data_loaded and np.sum(visit_counts) > 0:
    print("--- 正在生成三维访问密度图 ---")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 准备三维柱状图的数据
    x_data, y_data = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    x_data = x_data.flatten()
    y_data = y_data.flatten()
    
    # --- 【核心修改】将柱子的基座Z位置设置为指定的高度 ---
    z_data = np.full((GRID_SIZE * GRID_SIZE), BAR_BASE_HEIGHT)
    
    dx = dy = 0.8  
    
    # 获取原始的柱子高度数据并进行缩放
    dz_original = visit_counts.T.flatten()
    dz_scaled = dz_original / 2.0
    
    # 使用原始高度来计算颜色
    norm = plt.Normalize(dz_original.min(), dz_original.max())
    colors = cm.jet(norm(dz_original))

    # 绘制三维柱状图 (使用新的z_data作为起点)
    ax.bar3d(x_data, y_data, z_data, dx, dy, dz_scaled, color=colors, edgecolor='black')

    # 绘制底部的二维热力图 (位置仍在 z=0)
    X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
    ax.contourf(X, Y, visit_counts.T, zdir='z', offset=0, cmap='jet', alpha=0.5, norm=norm)

    # 设置图表样式
    #ax.set_title('3D State Visit Density', fontsize=18)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    #ax.set_zlabel('Visit Count (Height Scaled by 0.5)', fontsize=12)
    
    ax.invert_yaxis()
    
    # 添加颜色条
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.jet), ax=ax, shrink=0.6)
    cbar.set_label('Density of Visited States', fontsize=12)

    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    # 保存图表
    save_path = os.path.join(SAVE_DIR, 'visit_density_3d_raised.png')
    plt.savefig(save_path)
    print(f"图表 'visit_density_3d_raised.png' 已保存至: {save_path}")

else:
    print("没有有效数据可供绘图。")