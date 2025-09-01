from GridBoard import *
import numpy as np
from itertools import islice
import math
from energy_model import RobotEnergyModel 
# 1. <--- 新增这一行----------------------------



class Gridworld:

    def __init__(self, config, size=10, mode='static'):
        # super().__init__()
        self.config = config
        self.mode = config["mode"]
        self.size = config["grid_dimension"]
        if size >= 4:
            self.board = GridBoard(size=size)
        else:
            print("Minimum board size is 4. Initialized to size 4.")
            self.board = GridBoard(size=4)

        # Add pieces, positions will be updated later
        self.board.addPiece('Player', str(config["agent"]), eval(config["agent_pos"]))
        self.board.addPiece('Goal', str(config["exit"]), eval(config["exit_pos"]))
        # self.board.addPiece('Pit','-',(2,0))
        # self.board.addPiece('Wall','W',(1,0))
        # print(config[sources_pos])
        for ndx, (key, pos) in enumerate(config["sources_pos"]):
            self.board.addPiece('Source%s' % (ndx+1), key, pos)

        # 2. --- 新增下面的代码 ------------------------------
        self.energy_model = RobotEnergyModel(
            total_energy_capacity=config.get("total_energy_capacity", 1000.0),
            w_rot=config.get("w_rot", 2.0)
        )      
        # <--- 新增--------------------------------------------

        if mode == 'static':
            self.initGridStatic()
        elif mode == 'player':
            self.initGridPlayer()
        else:
            self.initGridRand()

    # Initialize stationary grid, all items are placed deterministically
    def initGridStatic(self):

        # Add pieces, positions will be updated later
        self.board.components['Player'].pos = self.config["agent_pos"]
        self.board.components['Goal'].pos = self.config["exit_pos"]
        # self.board.addPiece('Pit','-',(2,0))
        # self.board.addPiece('Wall','W',(1,0))
        for ndx, (key, pos) in enumerate(self.config["sources_pos"].items()):
            self.board.components['Source%s' % (ndx+1)].pos = pos




        # 3. --- 新增下面的代码 --------------------------
        if hasattr(self, 'energy_model'): # 确保 energy_model 已被创建
            self.energy_model.reset()
        self.board.components['Player'].heading = 90.0 # 假设初始朝向为0度
        # <--- 新增----------------------------------------







    # Check if board is initialized appropriately (no overlapping pieces)
    # also remove impossible-to-win boards
    def validateBoard(self):
        valid = True

        player = eval(self.board.components['Player'])
        goal = eval(self.board.components['Goal'])
        all_pos = [piece.pos for piece in self.board.components.items()]
        # print('all_pos:',all_pos)
        # all_positions = [player.pos, goal.pos, Source1.pos, Source2.pos, Source3.pos]
        if len(all_pos) > len(set(all_pos)):
            return False

        corners = [(0, 0), (0, self.board.size), (self.board.size, 0), (self.board.size, self.board.size)]
        # if player is in corner, can it move? if goal is in corner, is it blocked?
        if player.pos in corners or goal.pos in corners:
            val_move_pl = [self.validateMove('Player', addpos) for addpos in
                           [(0, 1), (1, 0), (-1, 0), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]]
            val_move_go = [self.validateMove('Goal', addpos) for addpos in
                           [(0, 1), (1, 0), (-1, 0), (0, -1), (-1, 1), (-1, -1), (1, 1), (1, -1)]]
            if 0 not in val_move_pl or 0 not in val_move_go:
                # print(self.display())
                # print("Invalid board. Re-initializing...")
                valid = False

        return valid

    # Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        # height x width x depth (number of pieces)
        self.initGridStatic()
        # place player
        self.board.components['Player'].pos = randPair(0, self.board.size)

        if not self.validateBoard():
            # print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    # Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        # height x width x depth (number of pieces)
        # self.board.components['Player'].pos = randPair(0,self.board.size)
        # self.board.components['Goal'].pos = randPair(0,self.board.size)
        self.board.components['Source1'].pos = randPair(0, self.board.size)
        # self.board.components['Source1'].pos = randPair(2,7)
        self.board.components['Source2'].pos = randPair(0, self.board.size)

        if not self.validateBoard():
            # print('Invalid grid. Rebuilding..')
            self.initGridRand()

    
    
    #4. --- 新增下面的代码 -----------------------------------
    def move(self, action):
        """
        根据动作原子性地更新Player的位置和朝向。
        这是确保朝向被正确追踪的关键。
        """
        player = self.board.components['Player']
        addpos_action = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, 1), (1, -1)]
        
        # 计算新位置
        move_vec = addpos_action[action]
        new_pos = addTuple(player.pos, move_vec)
        
        # 只有在移动有效时（没有撞墙），才更新位置和朝向
        if not self.board.check_if_collided(new_pos[0], new_pos[1]):
            player.pos = new_pos
            # 从能量模型中获取该动作对应的目标角度，并更新朝向
            target_angle, _ = self.energy_model.action_definitions[action]
            player.heading = target_angle
    # <--- 新增------------------------------------------------


    def get_reward(self, action, prev_heading, collided_or_stuck):

        player = self.board.components['Player']
        Ppos = player.pos
        Goal = self.board.components['Goal'].pos
        if type(Ppos) == str: Ppos = eval(Ppos)
        if type(Goal) == str: Goal = eval(Goal)

        # --- 1. 核心判断：任务是否因成功而结束 ---
        is_success = (Ppos == Goal) and not collided_or_stuck
        if is_success:
            # 如果成功到达终点，直接返回一个巨大的正奖励，不再计算其他项
            return 100.0

        # --- 2. 核心判断：任务是否因失败（撞墙/卡住）而结束 ---
        if collided_or_stuck:
            return self.config.get('penalty', -10.0) 

        # --- 3. 如果任务仍在进行中，计算稠密引导奖励 ---
        
        # a. 步数惩罚 (Time Penalty) - 鼓励走捷径
        time_penalty = -0.1

        # b. 辐射惩罚 (Radiation Penalty)
        radiation_penalty = 0
        RSq = 0
        for (i, d), (j, p) in zip(self.config['sources'].items(), self.config['sources_pos'].items()):
            dist_sq = ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
            RSq += d / (dist_sq + 1e-6) # 使用原始的 1/r^2，但增加稳定性

        # 增加一个权重来控制辐射惩罚的强度，我们称之为 w_rad
        w_rad = self.config.get('w_rad', 0.1) # 可以从 .yaml 配置，默认0.1
        radiation_penalty = - (w_rad * RSq) 

        # c. 能量惩罚 (Energy Penalty) - 我们的核心算法
        energy_cost = self.energy_model.get_step_cost(prev_heading, action)
        self.energy_model.update_energy(energy_cost)
        w_energy = self.config.get('w_energy', 0.1) # 从配置或默认值获取权重
        energy_penalty = - (w_energy * energy_cost)

        # d. 能量耗尽惩罚 - 这是一个特殊的失败条件
        if self.energy_model.is_depleted():
            print("EPISODE FAILED: OUT OF ENERGY")
            # 这是一个巨大的状态惩罚，而不是最终奖励
            # 我们会在 trainer.py 中通过 done=True 来处理它，并给予最终的惩罚
            pass

        # d. 【新增】距离奖励 (Distance Reward) - 由 w_dist 开关控制
        distance_reward = 0.0 # <--- 变量名修改为 reward
        w_dist = self.config.get('w_dist', 0.0) # 获取距离奖励权重，默认为0
        if w_dist > 0:
            dist_to_goal = math.sqrt((Ppos[0] - Goal[0])**2 + (Ppos[1] - Goal[1])**2)
            
            # 计算网格的最大可能距离（对角线）
            max_dist = math.sqrt(self.size**2 + self.size**2) 
            
            # 用最大距离减去当前距离，再乘以权重。这样离目标越近，奖励值越大。
            distance_reward = w_dist * (max_dist - dist_to_goal)

        # 将所有稠密奖励/惩罚相加
        step_reward = time_penalty + radiation_penalty + energy_penalty + distance_reward
        
        return step_reward

        #------------------------更改如上------------------------------





    def reward_dosefunc_1byr2_dm(self, dummy_move, vel=1, anticipated_pos=True):
        config = self.config
        pos_list = []
        for k, v in config['sources'].items():
            pos_list.append(k + 'pos')
        if anticipated_pos:
            Ppos = dummy_move
        else:
            Ppos = self.board.components['Player'].pos

        if type(Ppos) == str:
            Ppos = eval(Ppos)

        Goal = self.board.components['Goal'].pos
        if type(Goal) == str:
            Goal = eval(Goal)

        shift = 0.1
        RSq = 0
        for (i, d), (j, p) in zip(config['sources'].items(), config['sources_pos'].items()):
            # agent steps on source
            if Ppos[0] == p[0] and Ppos[1] == p[1]:
                RSq += d / (((Ppos[0] + shift) - p[0]) ** 2 + ((Ppos[1] + shift) - p[1]) ** 2)
                Gdist = (np.sqrt((Ppos[0] - Goal[0]) ** 2 + (Ppos[1] - Goal[1]) ** 2)) ** 2
            # agent reached exit
            elif Ppos == Goal:
                RSq += d / ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
                Gdist = 0
            else:
                RSq += d / ((Ppos[0] - p[0]) ** 2 + (Ppos[1] - p[1]) ** 2)
                Gdist = (np.sqrt((Ppos[0] - Goal[0]) ** 2 + (Ppos[1] - Goal[1]) ** 2)) ** 2

        if Ppos == Goal:
            Gdist = 0
            dose = (1 / vel) * RSq - (1 / (Gdist + shift))
        else:
            dose = (1 / vel) * RSq - (1 / Gdist)
        reward = -np.array(dose)
        return reward.item()

    def display(self):
        return self.board.render()
    

