import os
import carla
import pygame
import random
import numpy as np
import tensorflow as tf
import datetime
import json
from gym import spaces
from core.carla_function import VehicleFunction
from rl import utils
from rl import ThreeCameraCARLAEnvironment, CARLAEvent
from rl.environments.carla.tools import utils as carla_utils
from rl.environments.carla import env_utils

from typing import Dict, Tuple, Optional, Union


class CARLAEnv(ThreeCameraCARLAEnvironment):  # 暂定 还没修改
    ACTION = dict(space=spaces.Box(low=-1.0, high=1.0, shape=(2,)), default=np.zeros(shape=2, dtype=np.float32))

    VEHICLE_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(3,)),  # change 3 -- 4
                            default=np.zeros(shape=3, dtype=np.float32))

    NAVIGATION_FEATURES = dict()
    # ROAD_FEATURES = dict(space=spaces.Box(low=0.0, high=1.0, shape=(9,)), default=np.zeros(shape=9, dtype=np.float32))   # change

    # 待修改
    # VECTOR_MAP_FEATURES = dict(space=spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_LANES * POINTS_PER_LANE * 2,)),
    #                            default=np.zeros(shape=(MAX_LANES * POINTS_PER_LANE * 2,), dtype=np.float32))
    MAX_SURROUNDING_OBJECTS = 4
    FEATURES_PER_OBJECT = 9
    MAX_MAP_ELEMENTS = 4
    POINTS_PER_MAP_ELEMENT = 10
    FEATURES_PER_MAP_ELEMENT = 22

    VECTOR_MAP_SHAPE = (MAX_SURROUNDING_OBJECTS * FEATURES_PER_OBJECT) + \
                       (MAX_MAP_ELEMENTS * FEATURES_PER_MAP_ELEMENT)
    # 最终 UNIFIED_VECTOR_MAP_SHAPE 结果为 (4 * 9) + (4 * 22) = 36 + 88 = 124

    VECTOR_MAP_FEATURES = dict(
        space=spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(VECTOR_MAP_SHAPE,)  # <--- 修改: shape现在是 (124,)
        ),
        default=np.zeros(
            shape=(VECTOR_MAP_SHAPE,),  # <--- 修改: shape现在是 (124,)
            dtype=np.float32
        )
    )

    def __init__(self, *args, stack_depth=False, collision_penalty=10.0, info_every=1, time_horizon=4,
                 past_obs_freq=4, throttle_as_desired_speed=True, num_waypoints_for_feature=5,
                 range_controls: Optional[Dict[str, Tuple[float, float]]] = None, random_weathers: list = None,
                 random_towns: list = None, record_path: str = None, **kwargs):  # collision_penalty==100
        """
        :param stack_depth: if true the depth-image from the depth camera sensor will be stacked along the channel
                            dimension of the image, resulting in an image with an additional channel (e.g. 3 + 1 = 4)
        :param collision_penalty: how much the agent should be penalized for colliding with other objects.
        :param info_every: how frequently in terms of steps, the additional information should be gathered.
        :param range_controls: optional dict used to specify the range for each vehicle's control.
        :param time_horizon: how much observations to consider as a single one (suitable for RNN processing)
        :param past_obs_freq: how often (in terms of steps) to consider an observation as a past observation.
        :param num_waypoints_for_feature: how many waypoints to consider for the `navigation` feature vector.
        :param random_weathers: list of carla.WeatherParameters which are sampled at each environment reset.
        :param random_towns: list of town's names, which town is loaded at each environment reset.
        """
        assert info_every >= 1
        assert time_horizon >= 1
        assert past_obs_freq >= 1
        assert num_waypoints_for_feature >= 1

        # image_shape = kwargs.pop('image_shape', (90, 120, 3))  # change

        # if stack_depth:
        #     self.stack_depth = True
        #     image_shape = (image_shape[0], image_shape[1], image_shape[2] + 1)
        # else:
        #     self.stack_depth = False    # change

        # super().__init__(*args, image_shape=image_shape, **kwargs)   # change
        super().__init__(*args, **kwargs)

        self.penalty = collision_penalty
        self.next_waypoint = None
        self.info_every = info_every
        self.interpret_throttle_as_desired_speed = throttle_as_desired_speed

        # definition of `navigation` feature:   #change
        # self.num_waypoints = num_waypoints_for_feature
        # self.NAVIGATION_FEATURES['space'] = spaces.Box(low=0.0, high=25.0, shape=(self.num_waypoints,))
        # self.NAVIGATION_FEATURES['default'] = np.zeros(shape=self.num_waypoints, dtype=np.float32)
        self.NAVIGATION_FEATURES['space'] = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.NAVIGATION_FEATURES['default'] = np.zeros(shape=2, dtype=np.float32)

        self.last_llm_position = None
        # statistics
        self.episode = -1
        self.timestep = 0
        self.total_reward = 0.0
        self.success_count = 0

        self.range_controls = {} if range_controls is None else range_controls
        self.info_buffer = {k: [] for k in self.info_space.spaces.keys()}

        # time horizon and past obs:
        self.time_horizon = time_horizon
        self.past_obs_freq = past_obs_freq

        # init the past observations list with t empty (default) observations
        # NOTE: the last obs is always the current (most recent) one
        self.past_obs = self._init_past_obs()

        # Random weather:
        if isinstance(random_weathers, list):
            self.should_sample_weather = True
            self.weather_set = random_weathers

            for w in random_weathers:
                assert isinstance(w, carla.WeatherParameters)
        else:
            self.should_sample_weather = False

        # Random town:
        if random_towns is None:
            self.should_sample_town = False

        elif isinstance(random_towns, list):
            if len(random_towns) == 0:
                self.should_sample_town = False
            else:
                self.should_sample_town = True
                self.town_set = random_towns

        # Record (same images)
        if record_path is None:
            self.should_record = False
        else:
            self.should_record = True
            self.record_path = utils.makedir(record_path)

        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    'logs', 'stage-s1', current_time)
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化指标
        self.writer = tf.summary.create_file_writer(self.log_dir)

    # def define_sensors(self) -> dict:   # change
    #     from rl import SensorSpecs
    #     return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
    #                 imu=SensorSpecs.imu(),
    #                 front_camera=SensorSpecs.rgb_camera(position='on-top2', attachment_type='Rigid',
    #                                                              image_size_x=self.image_size[0],
    #                                                              image_size_y=self.image_size[1],
    #                                                              sensor_tick=self.tick_time),
    #                 left_camera=SensorSpecs.rgb_camera(position='lateral-left', attachment_type='Rigid',
    #                                                             image_size_x=self.image_size[0],
    #                                                             image_size_y=self.image_size[1],
    #                                                             sensor_tick=self.tick_time),
    #                 right_camera=SensorSpecs.rgb_camera(position='lateral-right', attachment_type='Rigid',
    #                                                              image_size_x=self.image_size[0],
    #                                                              image_size_y=self.image_size[1],
    #                                                              sensor_tick=self.tick_time))

    def define_sensors(self) -> dict:
        from rl import SensorSpecs
        return dict(collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    imu=SensorSpecs.imu(),
                    camera=SensorSpecs.rgb_camera(position='on-top2', attachment_type='Rigid',
                                                  image_size_x=120,
                                                  image_size_y=160,
                                                  sensor_tick=self.tick_time))

    @property
    def observation_space(self) -> spaces.Space:
        # return spaces.Dict(road=self.ROAD_FEATURES['space'], vehicle=self.VEHICLE_FEATURES['space'],
        #                    image=self.image_space, navigation=self.NAVIGATION_FEATURES['space'])   # change
        import ipdb
        # ipdb.set_trace()
        return spaces.Dict(vehicle=self.VEHICLE_FEATURES['space'],
                           navigation=self.NAVIGATION_FEATURES['space'], vector_map=self.VECTOR_MAP_FEATURES['space'])

    @property
    def info_space(self) -> spaces.Space:
        space: spaces.Dict = super().info_space

        return spaces.Dict(episode=spaces.Discrete(n=1), timestep=spaces.Discrete(n=1),
                           total_reward=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)),
                           reward=spaces.Box(low=-np.inf, high=np.inf, shape=(1,)), **space.spaces)

    def actions_to_control(self, actions):
        """Converts the given actions to vehicle's control"""
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0
        self.control.steer = float(actions[1])
        self.control.hand_brake = False
        self.control.reverse = False

        if self.interpret_throttle_as_desired_speed:
            desired_speed = (float(actions[0]) + 1.0) / 2
            desired_speed *= 12.0  # change
            current_speed = carla_utils.speed(self.vehicle)

            if current_speed == desired_speed:
                self.control.throttle = 0.0
                self.control.brake = 0.0

            elif current_speed > desired_speed:
                # brake
                self.control.throttle = 0.0
                self.control.brake = (current_speed - desired_speed) / 100.0
            else:
                # accelerate
                self.control.brake = 0.0
                self.control.throttle = (desired_speed - current_speed) / 100.0
        else:
            if carla_utils.speed(self.vehicle) < 10.0:
                self.control.brake = 0.0

        if 'throttle' in self.range_controls:
            throttle = self.range_controls['throttle']
            self.control.throttle = utils.clip(self.control.throttle, min_value=throttle[0], max_value=throttle[1])

        if 'brake' in self.range_controls:
            brake = self.range_controls['brake']
            self.control.brake = utils.clip(self.control.brake, min_value=brake[0], max_value=brake[1])

        if 'steer' in self.range_controls:
            steer = self.range_controls['steer']
            self.control.steer = utils.clip(self.control.steer, min_value=steer[0], max_value=steer[1])

    def reward(self, *args, respect_speed_limit=False, **kwargs) -> float:
        """Reward function"""
        speed = carla_utils.speed(self.vehicle)
        dw = self.route.distance_to_next_waypoint()

        if self.collision_penalty > 0.0:
            self.should_terminate = True
            return -self.collision_penalty

        if respect_speed_limit:
            speed_limit = self.vehicle.get_speed_limit()

            if speed > speed_limit:
                return speed_limit - speed

        r = speed * 0.5  # change self.similarity

        if r != 0.0:
            r /= max(1.0, (dw / 2.0) ** 2)

        return r

    #  proposed
    #     def reward(self, *args, respect_speed_limit=False, **kwargs) -> float:
    #         """Reward function with three components: safety/comfort/efficiency"""
    #         dw = self.route.distance_to_next_waypoint()
    #         cur_ego_loc = self.vehicle.get_location()

    #         should_call_llm = False
    #         if self.last_llm_position is None:
    #             should_call_llm = True
    #         else:
    #             distance_moved = self.last_llm_position.distance(cur_ego_loc)
    #             if distance_moved >= 1.5:  # 检查移动距离是否超过2米
    #                  should_call_llm = True

    #         if should_call_llm:
    #             print("LLM starts!!!")
    #             self.last_llm_position = cur_ego_loc  # 更新上次位置
    #             from codebook.automatic_gpt import monitor_script
    #             vector = monitor_script()
    #             self.vector = vector
    #                 #
    #         if not should_call_llm:
    #             vector = self.vector # 将 vector 存储为类的属性

    # # #         # 如果 vector 为 None 或者不符合条件，使用默认值
    #         if vector is None or len(vector) != 3:
    #             print('have not received vector !!!')
    #             vector = [0.4, 0.3, 0.3]  # 使用默认值
    #         speed = carla_utils.speed(self.vehicle)
    #         # dw = self.route.distance_to_next_waypoint()

    #         # 安全性奖励（防止碰撞）
    #         safety_reward = 10.0
    #         if self.collision_penalty > 0.0:
    #             self.should_terminate = True
    #             safety_reward = -self.collision_penalty
    #             return safety_reward

    #         # 舒适性奖励（控制加速度或者避免剧烈动作）
    #         acceleration = self.vehicle.get_acceleration().x
    #         comfort_reward = -abs(acceleration)  # 加速度越大越不舒服，取负值作为惩罚

    #         # 快捷性奖励（速度高且方向正确）
    #         r = speed * self.similarity
    #         if r != 0.0:
    #             r /= max(1.0, (dw / 2.0)**2)
    #         efficiency_reward = r

    #         # 总奖励 = 各部分奖励加权求和
    #         total_reward = (
    #             vector[0] * safety_reward +
    #             vector[1] * comfort_reward +
    #             vector[2] * efficiency_reward
    #         )

    #         return total_reward
    #     def reward(self, *args, respect_speed_limit=False, **kwargs) -> float:
    #         """Reward function with three components: safety/comfort/efficiency"""
    #         dw = self.route.distance_to_next_waypoint()
    #         cur_ego_loc = self.vehicle.get_location()

    #         should_call_llm = False
    #         if self.last_llm_position is None:
    #             should_call_llm = True
    #         else:
    #             distance_moved = self.last_llm_position.distance(cur_ego_loc)
    #             if distance_moved >= 1.5:  # 检查移动距离是否超过2米
    #                  should_call_llm = True

    #         if should_call_llm:
    #             print("LLM starts!!!")
    #             self.last_llm_position = cur_ego_loc  # 更新上次位置
    #             from codebook.automatic_gpt import monitor_script
    #             vector = monitor_script()
    #             self.vector = vector
    #                 #
    #         if not should_call_llm:
    #             vector = self.vector # 将 vector 存储为类的属性

    # # #         # 如果 vector 为 None 或者不符合条件，使用默认值
    #         if vector is None or len(vector) != 3:
    #             print('have not received vector !!!')
    #             vector = [0.4, 0.3, 0.3]  # 使用默认值
    #         speed = carla_utils.speed(self.vehicle)
    #         # dw = self.route.distance_to_next_waypoint()

    #         # 安全性奖励（防止碰撞）
    #         safety_reward = 10.0
    #         if self.collision_penalty > 0.0:
    #             self.should_terminate = True
    #             safety_reward = -self.collision_penalty
    #             return safety_reward

    #         # 舒适性奖励（控制加速度或者避免剧烈动作）
    #         acceleration = self.vehicle.get_acceleration().x
    #         comfort_reward = -abs(acceleration)  # 加速度越大越不舒服，取负值作为惩罚

    #         # 快捷性奖励（速度高且方向正确）
    #         r = speed * self.similarity
    #         if r != 0.0:
    #             r /= max(1.0, (dw / 2.0)**2)
    #         efficiency_reward = r

    #         # 总奖励 = 各部分奖励加权求和
    #         total_reward = (
    #             vector[0] * safety_reward +
    #             vector[1] * comfort_reward +
    #             vector[2] * efficiency_reward
    #         )

    #         return total_reward

    #         # Basic observations
    #         speed = carla_utils.speed(self.vehicle)
    #         # print('SPEED = ',speed)
    #         dw = self.route.distance_to_next_waypoint()
    #         cur_ego_loc = self.vehicle.get_location()

    # #         should_call_llm = False
    # #         if self.last_llm_position is None:
    # #             should_call_llm = True
    # #         else:
    # #             distance_moved = self.last_llm_position.distance(cur_ego_loc)
    # #             if distance_moved >= 1.5:  # 检查移动距离是否超过2米
    # #                 should_call_llm = True

    # #         if should_call_llm:
    # #             print("LLM starts!!!")
    # #             self.last_llm_position = cur_ego_loc  # 更新上次位置
    # #             from codebook.automatic_gpt import monitor_script
    # #             vector = monitor_script()
    # #             self.vector = vector

    # #         if not should_call_llm:
    # #             vector = self.vector # 将 vector 存储为类的属性

    # #         # 如果 vector 为 None 或者不符合条件，使用默认值
    # #         if vector is None or len(vector) != 3:
    # #             print('have not received vector !!!')
    #         # vector = [0.4, 0.3, 0.3]  # 使用默认值

    # #         vector = monitor_script()
    # #         if vector == 0  or len(vector) != 3:
    # #             vector = [0.4, 0.3, 0.3]

    #         vector = [0.4, 0.3, 0.3]   #   先test 用固定权重

    #         # ===================================
    #         # 1. Safety Reward (Negative Penalties)
    #         # ===================================
    #         safety_reward = 0.0

    #         # Collision detection (immediate termination)
    #         if self.collision_penalty > 0.0:
    #             self.should_terminate = True
    #             return  -1  #  safety_reward # -1  # Direct safety penalty

    #         if speed >= 3:
    #             safe_driving_bonus = 0.15 # 可以调整
    #         else:
    #             safe_driving_bonus = 0.07
    #         safety_reward += safe_driving_bonus
    #         speed_limit = self.vehicle.get_speed_limit()
    #         # Speed limit penalty (continuous)
    #         if respect_speed_limit:
    #             over_speed = max(0.0, speed - speed_limit)
    #             safety_reward -= min((over_speed / max_speed_penalty) * 0.3, 1.0)  # 归一化到 [-1,0]

    #         safety_reward = max(-1.0, min(safety_reward, 1.0))
    #         # print('Safety Reward = ', safety_reward)

    # ===================================
    # 2. Comfort Reward (Negative Penalties)
    # ===================================
    #         comfort_reward = 0.0

    #         # 获取纵向加速度（assume x 轴为纵向方向）
    #         acceleration = self.vehicle.get_acceleration().x
    #         # print('ACC = ', acceleration)

    #         # 归一化参数
    #         max_acceleration = 6.0  # 最大可接受加速度
    #         max_angular_velocity = 20  # 最大可接受角速度

    #         # 计算第一个惩罚项：超出舒适加速度的部分
    #         comfort_reward -= min((abs(acceleration) / max_acceleration) * 0.03, 1.0)

    #         # 获取角速度（假设 z 轴为横摆角速度）
    #         angular_velocity = self.vehicle.get_angular_velocity().z
    #         # print('angular_velocity = ', angular_velocity)

    #         # 计算第三个惩罚项：大角度转向惩罚
    #         comfort_reward -= min((abs(angular_velocity) / max_angular_velocity) * 0.03, 1.0)
    #         # 新增第三个惩罚项：速度不能太低（鼓励保持适当速度）
    #         min_comfort_speed = 5.0  # 设定最低舒适速度
    #         if speed < min_comfort_speed:
    #             low_speed_penalty = min((min_comfort_speed - speed) / min_comfort_speed, 1.0)  # 归一化
    #             comfort_reward -= low_speed_penalty * 0.1  # 给予一定的惩罚（调整权重）

    #         comfort_reward = max(-1.0, min(comfort_reward, 0.0))
    # #         # print('Comfort Reward = ', comfort_reward)

    # #         # ===================================
    # #         # 3. 效率奖励（仅低速惩罚）
    # #         # ===================================
    #         efficiency_reward = 0.0

    #         # 可调参数
    #         MIN_SPEED = 8.0          # 最低允许速度（m/s）
    #         MAX_PENALTY = 0.5        # 最大惩罚值
    #         PENALTY_WEIGHT = 0.3     # 惩罚系数
    #         MAX_RAW_PENALTY = 0.5    # 原始最大惩罚值（归一化前）
    #         REWARD_WEIGHT = 0.4       # 奖励系数

    #         HIGH_SPEED_THRESHOLD = 8.0

    #         if speed > HIGH_SPEED_THRESHOLD:
    #         # 计算超速比例（相对于阈值）
    #             over_speed_ratio = (speed - HIGH_SPEED_THRESHOLD) / HIGH_SPEED_THRESHOLD
    #             # 使用对数函数控制奖励增长
    #             import math
    #             reward = REWARD_WEIGHT * math.log(1 + over_speed_ratio) * 0.5
    #             efficiency_reward += min(reward, 0.1)

    #         # 核心惩罚逻辑
    #         if speed < MIN_SPEED:
    #             # 计算速度差比例
    #             speed_ratio = (MIN_SPEED - speed) / MIN_SPEED
    #             # 应用二次方惩罚（速度越低惩罚越重）
    #             penalty = PENALTY_WEIGHT * (speed_ratio ** 2)
    #             efficiency_reward -= min(penalty, MAX_PENALTY)

    #         # 归一化处理（将原始值映射到[-1,0]区间）
    #         normalized_efficiency = (efficiency_reward / MAX_RAW_PENALTY)* 0.4  # 除法保证最大惩罚-1
    #         efficiency_reward = max(-1.0, min(normalized_efficiency, 0.8))

    #         # 数值截断
    #         # print('efficiency_reward = ',efficiency_reward)

    #         # ===================================
    #         # Final Weighted Sum
    # ===================================
    #         total_reward = (
    #             vector[0] * safety_reward
    #             # +    # Safety weight
    #             # vector[1] * comfort_reward +   # Comfort weight
    #             # vector[2] * efficiency_reward  # Efficiency weight
    #         )
    #         # print('safety_reward = ',safety_reward)
    #         # print('comfort_reward = ',comfort_reward)
    #         # print('efficiency_reward = ',efficiency_reward)
    #         with open("total_reward.txt", "a") as f:
    #             f.write(f"{total_reward}\n")

    #         return total_reward

    #  Autoreward
    #     def reward_first_step(self, *args, respect_speed_limit=False, **kwargs) -> float:
    #         """Reward function with three components: safety/comfort/efficiency"""
    #         # Basic observations
    #         total_reward = 0
    #         speed = carla_utils.speed(self.vehicle)
    #         Vehicle_MAX_SPEED = 20
    #         # print('SPEED = ',speed)
    #         dw = self.route.distance_to_next_waypoint()

    #         # Collision detection (immediate termination)
    #         if self.collision_penalty > 0.0:
    #             self.should_terminate = True
    #             collision_penalty = -1.0 # -1  # Direct safety penalty
    #         else:
    #             collision_penalty = 0.0
    #         total_reward += collision_penalty

    #         speed_reward = speed/ Vehicle_MAX_SPEED
    #         total_reward += speed_reward

    #         right_lane_reward = self.similarity
    #         total_reward += right_lane_reward

    #         return total_reward

    #     def reward_second_step(self, *args, respect_speed_limit=False, **kwargs) -> float:
    #         """Reward function with three components: safety/comfort/efficiency"""
    #         # Basic observations
    #         total_reward = 0
    #         speed = carla_utils.speed(self.vehicle)
    #         Vehicle_MAX_SPEED = 20
    #         # print('SPEED = ',speed)
    #         dw = self.route.distance_to_next_waypoint()

    #         # Collision detection (immediate termination)
    #         if self.collision_penalty > 0.0:
    #             self.should_terminate = True
    #             collision_penalty = -1.0 # -1  # Direct safety penalty
    #         else:
    #             collision_penalty = 0.0
    #         total_reward += collision_penalty * 0.8

    #         speed_reward = speed/ Vehicle_MAX_SPEED
    #         total_reward += speed_reward * speed_reward * 1.5

    #         right_lane_reward = self.similarity
    #         total_reward += right_lane_reward * 2

    #         return total_reward
    #     def reward_second_step(self, *args, respect_speed_limit=False, **kwargs) -> float:
    #         """Reward function with three components: safety/comfort/efficiency"""
    #         # Basic observations
    #         total_reward = 0
    #         speed = carla_utils.speed(self.vehicle)
    #         Vehicle_MAX_SPEED = 20
    #         # print('SPEED = ',speed)
    #         dw = self.route.distance_to_next_waypoint()

    #         # Collision detection (immediate termination)
    #         if self.collision_penalty > 0.0:
    #             self.should_terminate = True
    #             collision_penalty = -1.0 # -1  # Direct safety penalty
    #         else:
    #             collision_penalty = 0.0
    #         total_reward += collision_penalty * 8

    #         speed_reward = speed/ Vehicle_MAX_SPEED
    #         total_reward += speed_reward * speed_reward * 5

    #         right_lane_reward = self.similarity
    #         total_reward += right_lane_reward * 2

    #         return total_reward

    def reset(self) -> dict:
        self.next_waypoint = None

        self.episode += 1
        self.timestep = 0
        self.total_reward = 0.0
        self.past_obs = self._init_past_obs()

        return super().reset()

    def reset_world(self):
        if self.should_sample_town:
            self.set_town(town=random.choice(self.town_set))

        if self.should_sample_weather:
            self.set_weather(weather=random.choice(self.weather_set))

        super().reset_world()

    def reset_info(self):
        for k in self.info_buffer.keys():
            self.info_buffer[k].clear()

    def render(self, *args, **kwargs):
        super().render()

        if self.should_record:
            pygame.image.save(self.display, os.path.join(self.record_path, f'{self.timestep}.jpeg'))

    def set_record_path(self, path):
        if isinstance(path, str):
            self.record_path = path
            self.should_record = True
        else:
            self.should_record = False
            self.record_path = None

    def step(self, actions):
        """Performs one environment step (i.e. it updates the world, etc.)"""
        state, reward, done, info = super().step(actions)

        if self.timestep % self.info_every == 0:
            for k, v in info.items():
                self.info_buffer[k].append(v)

        self.timestep += 1
        self.total_reward += reward

        return state, reward, done, info

    #     def on_collision(self, event: carla.CollisionEvent, **kwargs):
    #         actor_type = event.other_actor.type_id
    #         print(f'Collision with actor={actor_type})')
    #         self.trigger_event(event=CARLAEvent.ON_COLLISION, actor=actor_type)

    #         if 'pedestrian' in actor_type:
    #             self.collision_penalty += self.penalty
    #             self.should_terminate = True

    #         elif 'vehicle' in actor_type:
    #             self.collision_penalty += self.penalty / 2.0
    #             self.should_terminate = True
    #         else:
    #             self.collision_penalty += self.penalty / 100.0

    #         self.should_terminate = True
    def on_collision(self, event: carla.CollisionEvent, **kwargs):  # 检测碰撞
        actor_type = event.other_actor.type_id
        print(f'Collision with actor={actor_type})')
        self.trigger_event(event=CARLAEvent.ON_COLLISION, actor=actor_type)

        if 'pedestrian' in actor_type:
            self.collision_penalty += self.penalty
            self.should_terminate = True

        elif 'vehicle' in actor_type:
            self.collision_penalty += self.penalty  # / 2.0
            self.should_terminate = True
        else:
            self.collision_penalty += self.penalty  # / 100.0

        self.should_terminate = True

        # 记录碰撞信息
        self.success_count += 1
        self.log_metric('Metrics/Collisions', float(self.success_count), self.episode)

    def on_sensors_data(self, data: dict) -> dict:
        data = super().on_sensors_data(data)
        # change
        # if not self.stack_depth:
        # return data
        #         change
        #         # concatenate depth image along the channel axis
        #         depth = data['depth']
        #         r = depth[:, :, 0]
        #         g = depth[:, :, 1]
        #         b = depth[:, :, 2]

        #         depth = (r + g * 256 + b * 256 * 256) / (256**3 - 1)
        #         depth = np.log1p(depth * 1000.0)

        #         depth_image = np.concatenate([np.zeros_like(depth), depth, np.zeros_like(depth)], axis=1)
        #         data['camera'] = np.concatenate((data['camera'], depth_image), axis=-1)
        return data

    def get_observation(self, sensors_data: dict) -> Union[list, dict]:
        import ipdb
        # ipdb.set_trace()
        obs = self._get_observation(sensors_data)

        # consider an observation (over time) only at certain timesteps
        if self.timestep % self.past_obs_freq == 0:
            # update past observation list:
            self.past_obs.pop(0)  # remove the oldest (t=0)
            self.past_obs.append(obs)  # append the newest

        return self.past_obs.copy()

    #     def _get_observation(self, sensors_data: dict) -> dict:
    #         if len(sensors_data.keys()) == 0:
    #             # return default obs
    #             return dict(image=self.default_image, vehicle=self.VEHICLE_FEATURES['default'],
    #                         road=self.ROAD_FEATURES['default'], navigation=self.NAVIGATION_FEATURES['default'])

    #         # get image, reshape, and scale
    #         image = np.asarray(sensors_data['camera'], dtype=np.float32)

    #         if image.shape != self.image_shape:
    #             image = env_utils.resize(image, size=self.image_size)

    #         image /= 255.0

    #         # features
    #         vehicle_obs = self._get_vehicle_features()
    #         road_obs = self._get_road_features()
    #         navigation_obs = self._get_navigation_features()

    #         obs = dict(image=image, vehicle=vehicle_obs, road=road_obs, navigation=navigation_obs)
    #         return env_utils.replace_nans(obs)

    def _get_observation(self, sensors_data: dict) -> dict:  # 观测量 不带 SURROUNDING_FEATURES
        if len(sensors_data.keys()) == 0:
            # return default obs
            #  待加   周围交通流的特征
            # return dict(image=self.default_image, vehicle=self.VEHICLE_FEATURES['default'],
            #             road=self.ROAD_FEATURES['default'], navigation=self.NAVIGATION_FEATURES['default'])     # 运行时长过长  change
            return dict(vehicle=self.VEHICLE_FEATURES['default'],
                        navigation=self.NAVIGATION_FEATURES['default'],
                        vector_map=self.VECTOR_MAP_FEATURES['default'])  # 运行时长过长

        # get image, reshape, and scale
        # change
        #         image = np.asarray(sensors_data['camera'], dtype=np.float32)

        #         if image.shape != self.image_shape:
        #             image = env_utils.resize(image, size=self.image_size)

        #         image /= 255.0

        # features
        vehicle_obs = self._get_vehicle_features()
        # road_obs = self._get_road_features()  # change
        navigation_obs = self._get_navigation_features()
        vector_map_obs = self._get_vector_map_features()

        vehicle_location = self.vehicle.get_location()
        waypoint: carla.Waypoint = self.map.get_waypoint(self.vehicle.get_location())
        is_at_traffic_light = float(self.vehicle.is_at_traffic_light())

        vehicle_velocity = self.vehicle.get_velocity()

        # 存json文件 用来传给gpt用
        # 创建一个字典来存储 ego 信息
        ego_info = {
            "vehicle_location": {
                "x": vehicle_location.x,
                "y": vehicle_location.y,
                "z": vehicle_location.z
            },
            "vehicle_velocity": {
                "x": vehicle_velocity.x,
                "y": vehicle_velocity.y,
                "z": vehicle_velocity.z
            }
        }

        # 将 ego_info 字典转换为 JSON 格式并保存到文件
        # ego_info_path = "/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/ego_position.json"  # JSON 文件路径

        ego_info_path = "ego_position.json"
        with open(ego_info_path, 'w') as json_file:
            json.dump(ego_info, json_file, indent=4)  # 美化输出，缩进为 4 个空格
            # print("ego_position的json文件已经更新！！！！")

        # ipdb.set_trace()

        scene_info = {
            "scene_type": {
                "waypoint.is_intersection": float(waypoint.is_intersection),
                "waypoint.is_junction": float(waypoint.is_junction),
                "is_at_traffic_light": is_at_traffic_light
            }
        }
        # scene_info_path = "/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/scene_flag.json"  # JSON 文件路径
        scene_info_path = "scene_flag.json"
        with open(scene_info_path, 'w') as json_file:
            json.dump(scene_info, json_file, indent=4)  # 美化输出，缩进为 4 个空格

        # # 周围交通参与者的信息
        hero_vehicle = self.vehicle
        vehicle_function = VehicleFunction(self.world, self.vehicle)
        # ipdb.set_trace()
        previous_close_vehicles_info, surrounding_vehicles = vehicle_function.check_surrounding_vehicles(hero_vehicle)
        # print("previous_close_vehicles_info", previous_close_vehicles_info)

        #         surrounding_obs = self._get_surrounding_features(surrounding_vehicles)

        #         # 将信息存储到 JSON 文件中
        #         #surrounding_info_path = "/home/ubuntu/WorkSpacesPnCGroup/czw/My_recent_research/carla-driving-rl-agent-master/codebook/surrounding_positions.json"
        #         surrounding_info_path = "/root/czw_carla/My_recent_research/carla-driving-rl-agent-master/codebook/surrounding_positions.json"
        #         with open(surrounding_info_path, 'w') as json_file:
        #             json.dump(previous_close_vehicles_info, json_file, indent=4)

        # obs = dict(image=image, vehicle=vehicle_obs, road=road_obs, navigation=navigation_obs)   # change
        obs = dict(vehicle=vehicle_obs, navigation=navigation_obs, vector_map=vector_map_obs)
        return env_utils.replace_nans(obs)

    def _init_past_obs(self) -> list:
        """Returns a list of empty observations"""
        return [self._get_observation(sensors_data={}) for _ in range(self.time_horizon)]

    def get_info(self) -> dict:
        info = super().get_info()
        info['episode'] = self.episode
        info['timestep'] = self.timestep
        info['total_reward'] = self.total_reward

        # 681-691 新增
        # if not hasattr(self, 'step_counter'):
        #     self.step_counter = 0
        # self.step_counter += 1
        # if self.step_counter > 5000:
        #     # 使用新的reward计算方式
        #     reward = self.reward_second_step()
        #     if self.step_counter > 10000:  # 重置计数器
        #         self.step_counter = 0
        # else:
        # # 使用原始reward计算方式
        #     reward = self.reward_first_step()

        # 改成
        # reward = self.reward_first_step()
        reward = self.reward()

        info['reward'] = reward

        # info['reward'] = self.reward()
        return info

    def _get_road_features(self):
        """9 features:
            - 3: is_intersection, is_junction, is_at_traffic_light
            - 1: speed_limit
            - 5: traffic_light_state
        """
        waypoint: carla.Waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit = self.vehicle.get_speed_limit() / 100.0

        # Traffic light:
        is_at_traffic_light = float(self.vehicle.is_at_traffic_light())
        traffic_light_state = self.one_hot_traffic_light_state()

        return np.concatenate((
            [float(waypoint.is_intersection), float(waypoint.is_junction), is_at_traffic_light],
            [speed_limit],
            traffic_light_state), axis=0)

    def _get_vehicle_features(self):
        """4 features:
            - 1: similarity (e.g. current heading direction w.r.t. next route waypoint)
            - 1: speed
            - 1: throttle
            - 1: brake
        """
        return np.array([
            # self.similarity,    # change
            carla_utils.speed(self.vehicle) / 12.0,
            self.control.throttle,
            self.control.brake])

    def _get_navigation_features_old(self):
        """features: N distances from current vehicle location to N next route waypoints' locations
        """
        vehicle_location = self.vehicle.get_location()
        waypoints = self.route.get_next_waypoints(amount=self.num_waypoints)
        distances = []

        for w in waypoints:
            d = carla_utils.l2_norm(vehicle_location, w.transform.location) / self.num_waypoints
            distances.append(d)

        # pad the list with last (thus greater) distance if smaller then required
        if len(distances) < self.num_waypoints:
            for _ in range(self.num_waypoints - len(distances)):
                distances.append(distances[-1])

        return np.array(distances)

    def _get_navigation_features(self):
        """features: N distances from current vehicle location to N next route waypoints' locations
        改成 一个xy的坐标点
        """
        import ipdb
        # ipdb.set_trace()
        vehicle_location = self.vehicle.get_location()
        waypoints = self.route.get_next_waypoints(amount=2)
        # distances = []
        if len(waypoints) < 2:
            target_waypoint = waypoints[-1]
        else:
            target_waypoint = waypoints[1]  # 取前方第2个路点作为目标

        vehicle_transform = self.vehicle.get_transform()
        vec_to_waypoint = target_waypoint.transform.location - vehicle_location

        forward_vec = vehicle_transform.get_forward_vector()
        right_vec = vehicle_transform.get_right_vector()
        relative_x = vec_to_waypoint.x * forward_vec.x + vec_to_waypoint.y * forward_vec.y + vec_to_waypoint.z * forward_vec.z
        relative_y = vec_to_waypoint.x * right_vec.x + vec_to_waypoint.y * right_vec.y + vec_to_waypoint.z * right_vec.z

        return np.array([relative_x, relative_y], dtype=np.float32)

    def _get_vector_map_features(self):
        """
        核心函数：获取并打包所有特征到单一的vector_map中。

        :return: 一个形状为 (self.VECTOR_MAP_SIZE,) 的Numpy一维数组。
        """

        # 动态物体区的参数
        self.MAX_SURROUNDING_OBJECTS = 4  # <--- 修改: 6 -> 4
        # 每个物体的特征维度计算 (聚焦2D，简化类别)
        # 位置(2) + 尺寸(2) + 速度(2) + 偏航角(1) + 类别ID(1) + 有效位(1) = 9
        self.FEATURES_PER_OBJECT = 2 + 2 + 2 + 1 + 1 + 1  # <--- 修改: 13 -> 9
        self.OBJECTS_BLOCK_SIZE = self.MAX_SURROUNDING_OBJECTS * self.FEATURES_PER_OBJECT

        # 静态地图区的参数
        self.MAX_MAP_ELEMENTS = 4  # <--- 不变
        self.POINTS_PER_MAP_ELEMENT = 10  # <--- 修改: 20 -> 10，降低精度

        # 每个地图元素的特征维度计算 (几何点减少，简化类别)
        # 几何点(10*2) + 类别ID(1) + 有效位(1) = 22
        self.FEATURES_PER_MAP_ELEMENT = (self.POINTS_PER_MAP_ELEMENT * 2) + 1 + 1  # <--- 修改: 43 -> 22
        self.MAP_BLOCK_SIZE = self.MAX_MAP_ELEMENTS * self.FEATURES_PER_MAP_ELEMENT

        # 计算最终向量总长度 (结果将自动更新为 124)
        self.VECTOR_MAP_SIZE = self.OBJECTS_BLOCK_SIZE + self.MAP_BLOCK_SIZE

        # 初始化一个全为零的巨大向量
        vector_map = np.zeros(self.VECTOR_MAP_SIZE, dtype=np.float32)

        # 获取自车当前状态，用于坐标系变换
        ego_transform = self.vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())

        # --- 2. 填充动态物体信息区 (Dynamic Objects Block) ---

        # 获取所有车辆和行人actor
        actors = self.world.get_actors().filter('vehicle.*|walker.pedestrian.*')

        # 过滤掉自车和远处的物体
        nearby_actors = []
        for actor in actors:
            if actor.id == self.vehicle.id:
                continue
            if actor.get_location().distance(ego_transform.location) < 50.0:
                nearby_actors.append(actor)

        # 按距离排序，优先处理最近的物体
        nearby_actors.sort(key=lambda a: a.get_location().distance(ego_transform.location))

        for i, actor in enumerate(nearby_actors[:self.MAX_SURROUNDING_OBJECTS]):
            # 计算该物体特征向量的起始索引
            start_index = i * self.FEATURES_PER_OBJECT

            # 提取并转换数据
            actor_transform = actor.get_transform()

            # 位置 (2)
            pos_ego = self._world_to_ego_coords(...)[:2]  # <--- 修改: 只取x, y

            # 尺寸 (2)
            dims = np.array(
                [actor.bounding_box.extent.x * 2, actor.bounding_box.extent.y * 2])  # <--- 修改: 只取length, width

            # 速度 (2)
            vel_ego = self._world_to_ego_coords(..., is_vector=True)[:2]  # <--- 修改: 只取vx, vy

            # 偏航角 (1)
            yaw_ego = np.deg2rad(actor_transform.rotation.yaw - ego_transform.rotation.yaw)

            # 类别ID (1)
            class_id = 0.5 if 'vehicle' in actor.type_id else 1.0  # <--- 修改: 简化为单个ID

            # 有效位 (1)
            is_valid = 1.0

            # 拼接成该物体的特征向量
            object_features = np.concatenate([pos_ego, dims, vel_ego, [yaw_ego], [class_id], [is_valid]]).astype(
                np.float32)

            # 填入巨大的vector_map中
            vector_map[start_index: start_index + self.FEATURES_PER_OBJECT] = object_features

        # --- 3. 填充静态地图信息区 (Static Map Block) ---
        map_elements = self._extract_map_elements(ego_transform)

        for i, (element_type, points) in enumerate(map_elements[:self.MAX_MAP_ELEMENTS]):
            # 计算该地图元素特征向量的起始索引 (注意要加上物体区的总长度)
            start_index = self.OBJECTS_BLOCK_SIZE + (i * self.FEATURES_PER_MAP_ELEMENT)

            # 对点进行采样/填充至固定数量
            sampled_points = self._sample_or_pad_points(points, self.POINTS_PER_MAP_ELEMENT)

            # 扁平化坐标 (20*2=40)
            geom_features = sampled_points.flatten()

            # 类别ID (1)
            class_id = 0.5 if element_type == 'lane_divider' else 1.0  # <--- 修改: 简化为单个ID

            # 有效位 (1)
            is_valid = 1.0

            # 拼接成该地图元素的特征向量
            map_features = np.concatenate([geom_features, [class_id], [is_valid]]).astype(np.float32)

            # 填入巨大的vector_map中
            vector_map[start_index: start_index + self.FEATURES_PER_MAP_ELEMENT] = map_features

        return vector_map

    def _world_to_ego_coords(self, world_points, ego_matrix, is_vector=False):
        """辅助函数：将世界坐标转换为自车坐标"""
        ego_matrix_inv = np.linalg.inv(ego_matrix)

        # Reshape to (1, 3) if it's a single point/vector
        if world_points.ndim == 1:
            world_points = world_points.reshape(1, -1)

        # For vectors (like velocity), we don't apply translation
        if is_vector:
            ego_matrix_inv[0:3, 3] = 0.0

        points_h = np.hstack([world_points, np.ones((world_points.shape[0], 1))])
        ego_points_h = points_h @ ego_matrix_inv.T
        return ego_points_h[:, :3]

    def _extract_map_elements(self, ego_transform):
        """辅助函数：提取车道线和道路边界"""
        elements = []
        waypoint = self.map.get_waypoint(ego_transform.location)

        for i in range(-2, 3):  # 检查当前、左二、右二共5条车道
            wp = waypoint
            if i < 0:
                for _ in range(abs(i)): wp = wp.get_left_lane()
            elif i > 0:
                for _ in range(i): wp = wp.get_right_lane()

            if wp is None or wp.lane_type != carla.LaneType.Driving:
                continue

            # 提取车道中心线作为 'lane_divider'
            wps_center = wp.next_until_lane_end(40.0)  # 向前追溯40米
            if len(wps_center) > 1:
                points_center = self._wps_to_ego_points(wps_center, ego_transform)
                elements.append(('lane_divider', points_center))

            # 提取道路边界
            wps_boundary = wp.next_until_lane_end(40.0)
            if len(wps_boundary) > 1:
                points_boundary = self._wps_to_ego_points(wps_boundary, ego_transform, boundary='right')  # 取右边界
                elements.append(('road_boundary', points_boundary))

        return elements

    def _wps_to_ego_points(self, waypoints, ego_transform, boundary=None):
        """将carla路点列表转换为自车坐标系下的点集"""
        points_world = []
        for wp in waypoints:
            loc = wp.transform.location
            if boundary == 'right':
                loc += wp.transform.get_right_vector() * (wp.lane_width / 2)
            points_world.append([loc.x, loc.y])

        points_world = np.array(points_world)
        points_world_3d = np.c_[points_world, np.zeros(len(points_world))]
        points_ego = self._world_to_ego_coords(points_world_3d, np.array(ego_transform.get_matrix()))
        return points_ego[:, :2]

    def _sample_or_pad_points(self, points, num_points):
        """对点进行采样或填充"""
        if len(points) == num_points:
            return points

        # 点太少，用最后一个点填充
        if len(points) < num_points:
            padding = np.tile(points[-1], (num_points - len(points), 1))
            return np.vstack([points, padding])

        # 点太多，等距采样
        else:
            indices = np.linspace(0, len(points) - 1, num_points, dtype=int)
            return points[indices]

    def _update_target_waypoint(self):
        super()._update_target_waypoint()

        if self.next_waypoint is None:
            self.next_waypoint = self.route.next

        elif self.next_waypoint != self.route.next:
            self.next_waypoint = self.route.next

    def one_hot_traffic_light_state(self):
        if self.vehicle.is_at_traffic_light():
            state: carla.TrafficLightState = self.vehicle.get_traffic_light_state()
        else:
            state = carla.TrafficLightState.Unknown

        vector = np.zeros(shape=5, dtype=np.float32)
        vector[state] = 1.0
        return vector

    @staticmethod
    def one_hot_speed(speed: float):
        vector = np.zeros(shape=4, dtype=np.float32)

        if speed <= 30.0:
            vector[0] = 1.0
        elif 30.0 < speed <= 60.0:
            vector[1] = 1.0
        elif 60.0 < speed <= 90.0:
            vector[2] = 1.0
        else:
            # speed > 90.0
            vector[3] = 1.0

        return vector

    @staticmethod
    def one_hot_lane_change(lane: carla.LaneChange):
        vector = np.zeros(shape=4, dtype=np.float32)

        if lane is carla.LaneChange.NONE:
            vector[0] = 1.0
        elif lane is carla.LaneChange.Left:
            vector[1] = 1.0
        elif lane is carla.LaneChange.Right:
            vector[2] = 1.0
        else:
            # lane is carla.LaneChange.Both
            vector[3] = 1.0

        return vector

    @staticmethod
    def one_hot_lane_type(lane: carla.LaneType):
        vector = np.zeros(shape=5, dtype=np.float32)

        if lane is carla.LaneType.NONE:
            vector[0] = 1.0
        elif lane is carla.LaneType.Driving:
            vector[1] = 1.0
        elif lane is carla.LaneType.Sidewalk:
            vector[2] = 1.0
        elif lane is carla.LaneType.Stop:
            vector[3] = 1.0
        else:
            vector[4] = 1.0

        return vector

    @staticmethod
    def one_hot_lane_marking_type(lane: carla.LaneMarkingType):
        vector = np.zeros(shape=4, dtype=np.float32)

        if lane is carla.LaneMarkingType.NONE:
            vector[0] = 1.0
        elif lane is carla.LaneMarkingType.Broken:
            vector[1] = 1.0
        elif lane is carla.LaneMarkingType.Solid:
            vector[2] = 1.0
        else:
            vector[3] = 1.0

        return vector

    def one_hot_similarity(self, threshold=0.3):
        vector = np.zeros(shape=4, dtype=np.float32)

        if self.similarity > 0.0:
            if self.similarity >= 1.0 - threshold:
                vector[0] = 1.0
            else:
                vector[1] = 1.0
        else:
            if self.similarity <= threshold - 1.0:
                vector[2] = 1.0
            else:
                vector[3] = 1.0

        return vector

    @staticmethod
    def one_hot_waypoint_distance(distance: float, very_close=1.5, close=3.0):
        vector = np.zeros(shape=3, dtype=np.float32)

        if distance <= very_close:
            vector[0] = 1.0
        elif very_close < distance <= close:
            vector[1] = 1.0
        else:
            vector[2] = 1.0

        return vector

    def log_metric(self, name, value, step=None):
        """通用的指标记录方法

        Args:
            name (str): 指标名称
            value (float): 指标值
            step (int, optional): 步数，如果不指定则使用 total_steps
        """
        if step is None:
            step = self.total_steps

        with self.writer.as_default():
            try:
                tf.summary.scalar(name, float(value), step=step)
                self.writer.flush()
            except Exception as e:
                print(f"Error writing metric {name} to TensorBoard: {e}")
                print(traceback.format_exc())