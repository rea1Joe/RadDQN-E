# energy_model.py
import math

class RobotEnergyModel:
    """
    一个用于计算机器人能耗的模块化类。
    它采用无量纲的相对成本模型，并追踪累计的能量消耗。
    """
    def __init__(self, total_energy_capacity=1000.0, w_rot=2.0):
        self.w_rot = w_rot
        # 这个动作映射关系与trainer.py中的addpos_action完全对应
        # 动作索引 -> (目标角度, 移动距离)
        self.action_definitions = {
            0: (270, 1.0),      # 上 (-1, 0)
            1: (90, 1.0),       # 下 (1, 0)
            2: (180, 1.0),      # 左 (0, -1)
            3: (0, 1.0),        # 右 (0, 1)
            4: (315, math.sqrt(2)), # 左上 (-1, 1)
            5: (225, math.sqrt(2)), # 左下 (-1, -1)
            6: (45, math.sqrt(2)),  # 右上 (1, 1)
            7: (135, math.sqrt(2))  # 右下 (1, -1)
        }
        self.total_energy_capacity = total_energy_capacity
        self.remaining_energy = total_energy_capacity
        
    def _calculate_angle_diff(self, angle1, angle2):
        """计算两个角度之间的最小差值（0-180度之间）。"""
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)

    def get_step_cost(self, current_heading, action):
        """计算执行单个动作的瞬时相对成本 (ΔC_t)。"""
        if action not in self.action_definitions:
            return 0.0

        target_angle, distance_delta = self.action_definitions[action]
        angle_diff = self._calculate_angle_diff(current_heading, target_angle)
        
        rotation_cost = self.w_rot * (angle_diff / 180.0)
        translation_cost = distance_delta
        
        return rotation_cost + translation_cost

    def update_energy(self, cost):
        """根据单步成本更新剩余能量。"""
        self.remaining_energy -= cost

    def is_depleted(self):
        """检查电池能量是否耗尽。"""
        return self.remaining_energy <= 0

    def reset(self):
        """重置电池状态。"""
        self.remaining_energy = self.total_energy_capacity