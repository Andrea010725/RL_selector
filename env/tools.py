import sys

import carla
import random
import numpy as np
import pickle
import os
import xml.etree.ElementTree as ET
from functools import reduce
import re
import math
import time
from datetime import datetime
import json



VEHICLE_TYPE_DICT = {
    'vehicle.audi.a2': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.audi.etron': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.audi.tt': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.bmw.grandtourer': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.chevrolet.impala': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.citroen.c3': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.dodge_charger.police': ['police'],  # special scene
    'vehicle.jeep.wrangler_rubicon': ['car', 'suv', 'wheel4', 'common', 'czw1'],
    'vehicle.lincoln.mkz_2020': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.mercedes.coupe': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.mini.cooper_s': ['car', 'wheel4', 'common'],
    'vehicle.nissan.micra': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.nissan.patrol': ['car', 'suv', 'wheel4', 'common', 'czw1'],
    'vehicle.seat.leon': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.toyota.prius': ['car', 'wheel4', 'common', 'czw1'],
    'vehicle.carlamotors.carlacola': ['truck', 'large', 'wheel4', 'common', 'czw1'],
    'vehicle.tesla.cybertruck': ['truck', 'large', 'wheel4', 'common', 'czw1'],
    'vehicle.volkswagen.t2': ['bus', 'large', 'wheel4', 'common', 'czw1'],
    'vehicle.harley-davidson.low_rider': ['moto', 'wheel2', 'common'],
    'vehicle.kawasaki.ninja': ['moto', 'wheel2', 'common'],
    'vehicle.yamaha.yzf': ['moto', 'wheel2', 'common'],
    'vehicle.bh.crossbike': ['bicycle', 'wheel2'],
    'vehicle.diamondback.century': ['bicycle', 'wheel2'],
    'vehicle.gazelle.omafiets': ['bicycle', 'wheel2'],
}
TYPE_VEHICLE_DICT = {}
for bp_name_outside, bp_filters_outside in VEHICLE_TYPE_DICT.items():
    for bp_filter_outside in bp_filters_outside:
        if bp_filter_outside not in TYPE_VEHICLE_DICT:
            TYPE_VEHICLE_DICT[bp_filter_outside] = []
        TYPE_VEHICLE_DICT[bp_filter_outside].append(bp_name_outside)

OBSTACLE_TYPE_DICT = {
    # traffic obstacles
    'static.prop.garbage01': ['garbage'],  # 建筑垃圾
    'static.prop.garbage02': ['garbage'],
    'static.prop.garbage03': ['garbage'],
    'static.prop.garbage04': ['garbage'],
    'static.prop.busstop ': ['bus_stop'],  # 公交车站
    'static.prop.constructioncone': ['construction'],  # 施工锥，用于标记施工区域或指引行人和车辆
    'static.prop.streetbarrier': ['street_barrier'],  # 用于限制车辆通行或指引行人。
    'static.prop.trafficwarning': ['street_barrier'],
    'static.prop.trafficcone01': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'static.prop.trafficcone02': ['traffic_barrier'],  # 交通锥，用于标记道路施工区域或指引交通
    'walker.pedestrian.0004': ['workers'],
    'walker.pedestrian.0003': ['workers'],
    'walker.pedestrian.0015': ['workers'],
    'walker.pedestrian.0009': ['workers'],
    'walker.pedestrian.0006': ['workers'],
    'walker.pedestrian.0013': ['workers'],
    'static.prop.creasedbox02': ['creasedbox'],
    'static.prop.ironplank': ['plank']
}
TYPE_OBSTACLE_DICT = {}
for bp_obstacle_name, bp_obstacle_filters in OBSTACLE_TYPE_DICT.items():
    for bp_obstacle_filters in bp_obstacle_filters:
        if bp_obstacle_filters not in TYPE_OBSTACLE_DICT:
            TYPE_OBSTACLE_DICT[bp_obstacle_filters] = []
        TYPE_OBSTACLE_DICT[bp_obstacle_filters].append(bp_obstacle_name)

WALKER_TYPE_DICT = {
    'walker.pedestrian.0004': ['workers'],
    'walker.pedestrian.0003': ['workers'],
    'walker.pedestrian.0015': ['workers'],
    'walker.pedestrian.0019': ['workers'],
    'walker.pedestrian.0016': ['workers'],
    'walker.pedestrian.0023': ['workers'],
}
TYPE_WALKER_DICT = {}
for bp_WALKER_name, bp_WALKER_filters in WALKER_TYPE_DICT.items():
    for bp_WALKER_filters in bp_WALKER_filters:
        if bp_WALKER_filters not in TYPE_WALKER_DICT:
            TYPE_WALKER_DICT[bp_WALKER_filters] = []
        TYPE_WALKER_DICT[bp_WALKER_filters].append(bp_WALKER_name)

SUV = ['vehicle.audi.etron',
       'vehicle.nissan.patrol',
       'vehicle.nissan.patrol_2021']
TRUCK = [
    'vehicle.carlamotors.carlacola',
    'vehicle.tesla.cybertruck']
LARGE_VEHICLE = SUV + TRUCK




class SceneManager():
    def __init__(self, wp, idp):
        self.origin_type = 'designation'  # 'map' means sample a random point from the world's map
        self.origin = None
        self.origin_location = None
        self.selected_task = None
        self.destination_type = 'map'
        self.initialized = False
        self.init_speed = 20
        self.navigation_cmds = ['Straight']
        self.current_ego_speed = -1
        self.last_llm_position = None
        self.passed_lane_ids = []
        self.stage = 0
        self.lane_changed = False
        self.idp = idp
        self.wp = wp
        # 新增：用于管理生成的CARLA对象
        self.generated_actors = []


    def _traffic_flow_scenario(self, filters='+common', forward_num=6, backward_num=4, **kwargs):  # idp=0.5
        # Desc: 在当前waypoint的左侧车道或者右侧车道生成车流
        results = []

        # Desc: 先向前生成车流
        _vehicle_wp = self.wp
        right_forward_index = 1
        while right_forward_index <= forward_num:
            bp_name = self.choose_bp_name(filters)
            if random.random() < self.idp:
                _vehicle_wp_new = carla.Transform(
                    location=carla.Location(x=_vehicle_wp.transform.location.x, y=_vehicle_wp.transform.location.y,
                                            z=_vehicle_wp.transform.location.z + 1),
                    rotation=_vehicle_wp.transform.rotation)
                results.append((bp_name, _vehicle_wp_new))
            _vehicle_wps = _vehicle_wp.next(random.randint(3, 25))
            if len(_vehicle_wps) == 0:
                break
            _vehicle_wp = _vehicle_wps[0]
            right_forward_index += 1

        # Desc: 再向后生成车流
        _vehicle_wp = self.wp
        right_backward_index = 1
        while right_backward_index <= backward_num:
            _vehicle_wps = _vehicle_wp.previous(8)
            _vehicle_wp_new = carla.Transform(
                location=carla.Location(x=_vehicle_wp.transform.location.x, y=_vehicle_wp.transform.location.y,
                                        z=_vehicle_wp.transform.location.z + 1),
                rotation=_vehicle_wp.transform.rotation)

            if len(_vehicle_wps) == 0:
                break
            _vehicle_wp = _vehicle_wps[0]
            bp_name = self.choose_bp_name(filters)
            if random.random() < self.idp:
                results.append((bp_name, _vehicle_wp_new))
            right_backward_index += 1

        return results

    def choose_bp_name(self, filters):
        """
        Desc: 根据车辆类型和车轮数选择对应的blueprint
        @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
        """
        # Special: 类型说明
        # car: 轿车
        # suv: SUV
        # truck: 卡车
        # van: 箱型车
        # bus: 巴士
        # moto: 摩托车
        # electric: 电瓶车
        # bicycle: 自行车
        # special: 特种车辆
        # police: 警车
        # fire: 消防车
        # wheel2: 两轮车辆
        # wheel4: 四轮车辆
        # large: 大型车辆
        # small: 小型车辆
        # common: 常见车辆：排除了特种车辆和自行车和小型车辆

        # e.g. +wheel4-special
        filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

        # 不能为单数
        if len(filters) % 2 != 0:
            return ""

        candidate_bp_names = []
        for index in range(0, len(filters), 2):
            op = filters[index]
            filter_type = filters[index + 1]
            if op == '+':
                candidate_bp_names.extend(TYPE_VEHICLE_DICT[filter_type])
            elif op == '-':
                candidate_bp_names = list(set(candidate_bp_names) - set(TYPE_VEHICLE_DICT[filter_type]))
            else:
                print(f'Error: {op} is not supported in blueprint choosing.')
                return ""

        if len(candidate_bp_names) == 0:
            print(f'Error: candidate_bp_names is empty.')
            return ""

        return random.choice(candidate_bp_names)

    def _apply_bp_generation(self, world, bp_and_transforms, name_prefix='vehicle'):
        offset_index = 0
        for v_index, (v_bp, v_transform) in enumerate(bp_and_transforms):
            v_bp = world.get_blueprint_library().find(v_bp)
            try:
                right_actor = world.spawn_actor(v_bp, v_transform)
                # 新增：将生成的actor添加到追踪列表中
                self.generated_actors.append(right_actor)
                if right_actor.type_id.startswith('vehicle'):
                    right_actor.set_autopilot(enabled=True)
                    # right_actor.set_target_velocity(random.randint(10, 15))   要寫三維的形式
                    right_actor.apply_control(carla.VehicleControl(throttle=(random.randint(3, 7) / 10)))
                else:
                    offset_index += 1
                continue
            except RuntimeError as e:
                if "collision at spawn position" in str(e):
                    continue
                else:
                    raise

    def right_traffic_flow_scenario(self, world, scene_cfg=None, gen_cfg=None):
            """
            修正版：在 self.wp 的右侧可行驶车道上生成交通流。
            使用 try...finally 确保 self.wp 在函数结束时被恢复。
            """
            if scene_cfg is None: scene_cfg = {}
            if gen_cfg is None: gen_cfg = {}

            original_wp = self.wp  # 保存原始状态
            try:
                processed_lanes = {self.wp.lane_id}
                driving_lane_count = 0

                while True:
                    next_wp = self.wp.get_right_lane()
                    if next_wp is None or next_wp.lane_id * self.wp.lane_id < 0:
                        break

                    self.wp = next_wp  # 临时修改 self.wp 用于迭代

                    if self.wp.lane_type != carla.LaneType.Driving or self.wp.lane_id in processed_lanes:
                        continue
                    if driving_lane_count >= scene_cfg.get('lane_num', 999):
                        break

                    bp_and_transforms = self._traffic_flow_scenario(**scene_cfg)
                    self._apply_bp_generation(world, bp_and_transforms, **gen_cfg)
                    processed_lanes.add(self.wp.lane_id)
                    driving_lane_count += 1
            finally:
                self.wp = original_wp  # 无论如何，恢复原始状态

    def left_traffic_flow_scenario(self, world, scene_cfg=None, gen_cfg=None):
            """
            修正版：在 self.wp 的左侧可行驶车道上生成交通流。
            """
            if scene_cfg is None: scene_cfg = {}
            if gen_cfg is None: gen_cfg = {}

            original_wp = self.wp  # 保存原始状态
            try:
                processed_lanes = {self.wp.lane_id}
                driving_lane_count = 0

                while True:
                    next_wp = self.wp.get_left_lane()
                    if next_wp is None or next_wp.lane_id * self.wp.lane_id < 0:
                        break

                    self.wp = next_wp  # 临时修改 self.wp

                    if self.wp.lane_type != carla.LaneType.Driving or self.wp.lane_id in processed_lanes:
                        continue
                    if driving_lane_count >= scene_cfg.get('lane_num', 999):
                        break

                    bp_and_transforms = self._traffic_flow_scenario(**scene_cfg)
                    self._apply_bp_generation(world, bp_and_transforms, **gen_cfg)
                    processed_lanes.add(self.wp.lane_id)
                    driving_lane_count += 1
            finally:
                self.wp = original_wp  # 恢复原始状态



    def right_parking_vehicle_scenario(self, world, wp, scene_cfg=None, gen_cfg=None):
            if scene_cfg is None: scene_cfg = {}
            if gen_cfg is None: gen_cfg = {}

            processed_lanes = set()
            stop_lane_count = 0
            current_wp = wp

            while True:
                is_parkable = current_wp.lane_type == carla.LaneType.Parking or current_wp.lane_type == carla.LaneType.Shoulder
                if is_parkable and current_wp.lane_id not in processed_lanes:

                    original_wp = self.wp
                    self.wp = current_wp
                    bp_and_transforms = self._traffic_flow_scenario(**scene_cfg)
                    self.wp = original_wp

                    self._apply_bp_generation(world, bp_and_transforms, **gen_cfg)
                    processed_lanes.add(current_wp.lane_id)
                    stop_lane_count += 1
                    if stop_lane_count >= scene_cfg.get('lane_num', 999):
                        break

                next_wp = current_wp.get_right_lane()
                if next_wp is None: break
                current_wp = next_wp

    def gen_traffic_flow(self, world, ego_wp):
            """
            总调用函数 - 保持原始调用接口不变
            """
            # 在执行任何操作前，确保类的状态 self.wp 是正确的
            self.wp = ego_wp

            # For: 右侧交通流
            self.right_traffic_flow_scenario(world,
                                             scene_cfg={'filters': '+czw1', 'idp': self.idp, 'lane_num': 2},
                                             gen_cfg={'name_prefix': 'right'})
            # For: 左侧交通流
            self.left_traffic_flow_scenario(world,
                                            scene_cfg={'filters': '+czw1', 'idp': self.idp, 'lane_num': 2},
                                            gen_cfg={'name_prefix': 'left'})
            # For: 对向交通流
            # self.opposite_traffic_flow_scenario(world,
            #                                     scene_cfg={'filters': '+czw1', 'idp': self.idp, 'lane_num': 2,
            #                                                'backward_num': 2},
            #                                     gen_cfg={'name_prefix': 'opposite'})
            # For: 路边停靠车辆 (此函数按原样传递ego_wp)
            # self.right_parking_vehicle_scenario(world, ego_wp,
            #                                     scene_cfg={'filters': '+wheel4-large', 'idp': self.idp,
            #                                                'forward_num': 2, 'lane_num': 1},
            #                                     gen_cfg={'name_prefix': 'park'})
            #


    def gen_traffic_flow_low_density(self, world, ego_wp):
        """低密度交通流生成"""
        # 低生成概率：0.1-0.3
        random_perc = random.randint(1, 3) / 10

        # 右侧交通流 - 减少车道数和车辆数
        self.right_traffic_flow_scenario(world,
                                         scene_cfg={
                                             'filters': '+czw1',
                                             'idp': random_perc,
                                             'lane_num': 1,  # 只影响1个车道
                                             'forward_num': 3,  # 前方3辆车
                                             'backward_num': 2  # 后方2辆车
                                         },
                                         gen_cfg={'name_prefix': 'right'})

        # 左侧交通流
        self.left_traffic_flow_scenario(world,
                                        scene_cfg={
                                            'filters': '+czw1',
                                            'idp': random_perc,
                                            'lane_num': 1,
                                            'forward_num': 3,
                                            'backward_num': 2
                                        },
                                        gen_cfg={'name_prefix': 'left'})

        # 对向交通流 - 减少车辆数
        # self.opposite_traffic_flow_scenario(world,
        #                                     scene_cfg={
        #                                         'filters': '+czw1',
        #                                         'idp': random_perc,
        #                                         'backward_num': 1  # 只生成1辆对向车
        #                                     },
        #                                     gen_cfg={'name_prefix': 'opposite'})

        # 停车车辆 - 很少
        self.right_parking_vehicle_scenario(world, ego_wp,
                                            scene_cfg={
                                                'filters': '+wheel4-large',
                                                'idp': random_perc * 0.5,  # 更低概率
                                                'forward_num': 1
                                            },
                                            gen_cfg={'name_prefix': 'park'})

    def gen_traffic_flow_medium_density(self, world, ego_wp):
        """中密度交通流生成"""
        # 中等生成概率：0.4-0.6
        random_perc = random.randint(4, 6) / 10

        # 右侧交通流
        self.right_traffic_flow_scenario(world,
                                         scene_cfg={
                                             'filters': '+czw1',
                                             'idp': random_perc,
                                             'lane_num': 2,  # 影响2个车道
                                             'forward_num': 6,  # 前方6辆车
                                             'backward_num': 4  # 后方4辆车
                                         },
                                         gen_cfg={'name_prefix': 'right'})

        # 左侧交通流
        self.left_traffic_flow_scenario(world,
                                        scene_cfg={
                                            'filters': '+czw1',
                                            'idp': random_perc,
                                            'lane_num': 2,
                                            'forward_num': 6,
                                            'backward_num': 4
                                        },
                                        gen_cfg={'name_prefix': 'left'})

        # # 对向交通流
        # self.opposite_traffic_flow_scenario(world,
        #                                     scene_cfg={
        #                                         'filters': '+czw1',
        #                                         'idp': random_perc,
        #                                         'backward_num': 3  # 3辆对向车
        #                                     },
        #                                     gen_cfg={'name_prefix': 'opposite'})

        # 停车车辆
        self.right_parking_vehicle_scenario(world, ego_wp,
                                            scene_cfg={
                                                'filters': '+wheel4-large',
                                                'idp': random_perc,
                                                'forward_num': 2
                                            },
                                            gen_cfg={'name_prefix': 'park'})

    def gen_traffic_flow_high_density(self, world, ego_wp):
        """高密度交通流生成"""
        # 高生成概率：0.7-0.9
        random_perc = random.randint(7, 9) / 10

        # 右侧交通流
        self.right_traffic_flow_scenario(world,
                                         scene_cfg={
                                             'filters': '+czw1',
                                             'idp': random_perc,
                                             'lane_num': 3,  # 影响3个车道
                                             'forward_num': 10,  # 前方10辆车
                                             'backward_num': 8  # 后方8辆车
                                         },
                                         gen_cfg={'name_prefix': 'right'})

        # 左侧交通流
        self.left_traffic_flow_scenario(world,
                                        scene_cfg={
                                            'filters': '+czw1',
                                            'idp': random_perc,
                                            'lane_num': 3,
                                            'forward_num': 10,
                                            'backward_num': 8
                                        },
                                        gen_cfg={'name_prefix': 'left'})

        # 对向交通流
        # self.opposite_traffic_flow_scenario(world,
                                            # scene_cfg={
                                            #     'filters': '+czw1',
                                            #     'idp': random_perc,
                                            #     'backward_num': 5  # 5辆对向车
                                            # },
                                            # gen_cfg={'name_prefix': 'opposite'})

        # 停车车辆 - 更多
        self.right_parking_vehicle_scenario(world, ego_wp,
                                            scene_cfg={
                                                'filters': '+wheel4-large',
                                                'idp': random_perc,
                                                'forward_num': 4
                                            },
                                            gen_cfg={'name_prefix': 'park'})

    def gen_traffic_flow_by_density(self, world, ego_wp, density_level="medium"):
        """根据密度级别生成交通流

        Args:
            world: CARLA世界对象
            ego_wp: 自车waypoint
            density_level: 密度级别 ("low", "medium", "high")
        """
        if density_level == "low":
            print(f"生成低密度交通流...")
            self.gen_traffic_flow_low_density(world, ego_wp)
        elif density_level == "medium":
            print(f"生成中密度交通流...")
            self.gen_traffic_flow_medium_density(world, ego_wp)
        elif density_level == "high":
            print(f"生成高密度交通流...")
            self.gen_traffic_flow_high_density(world, ego_wp)
        else:
            print(f"未知密度级别 '{density_level}'，使用默认中密度")
            self.gen_traffic_flow_medium_density(world, ego_wp)



    def ahead_obstacle_scenario(self, world, wp, scene_cfg={}, gen_cfg={}):  # construction 1 or construction 2
        # 检查gen_cfg字典中是否有name_prefix键，根据其值决定调用哪个函数
        name_prefix = gen_cfg.get('gen_cfg', 'construction1')  # 默认值为construction1
        if name_prefix == 'construction1':
            return self.ahead_obstacle_scenario_first(world, wp, scene_cfg, gen_cfg)
        elif name_prefix == 'construction2':
            return self.ahead_obstacle_scenario_second(world, wp, scene_cfg, gen_cfg)
        else:
            raise ValueError(
                "Invalid generation configuration. 'name_prefix' must be 'construction1' or 'construction2'.")

    def choose_obsbp_name(self, filters):
        """
        Desc: 根据障碍物类型选择对应的blueprint
        @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
        """
        # garbage: 道路垃圾，废弃物
        # bus_stop: 公交车站
        # construction： 施工
        # street_barrier: 道路指引
        # traffic_barrier: 交通障碍物

        filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

        # 不能为单数
        if len(filters) % 2 != 0:
            return ""

        candidate_obsbp_names = []
        for index in range(0, len(filters), 2):
            op = filters[index]
            filter_type = filters[index + 1]
            if op == '+':
                candidate_obsbp_names.extend(TYPE_OBSTACLE_DICT[filter_type])
            elif op == '-':
                candidate_obsbp_names = list(set(candidate_obsbp_names) - set(TYPE_OBSTACLE_DICT[filter_type]))
            else:
                print(f'Error: {op} is not supported in blueprint choosing.')
                return ""

        if len(candidate_obsbp_names) == 0:
            print(f'Error: candidate_bp_names is empty.')
            return ""

        return random.choice(candidate_obsbp_names)

    def choose_walker_name(self, filters):
        """
        Desc: 根据障碍物类型选择对应的blueprint
        @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
        @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
        """
        filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

        # 不能为单数
        if len(filters) % 2 != 0:
            return ""

        candidate_WALKERbp_names = []
        for index in range(0, len(filters), 2):
            op = filters[index]
            filter_type = filters[index + 1]
            if op == '+':
                candidate_WALKERbp_names.extend(TYPE_WALKER_DICT[filter_type])
            elif op == '-':
                candidate_WALKERbp_names = list(set(candidate_WALKERbp_names) - set(TYPE_WALKER_DICT[filter_type]))
            else:
                print(f'Error: {op} is not supported in blueprint choosing.')
                return ""

        if len(candidate_WALKERbp_names) == 0:
            print(f'Error: candidate_bp_names is empty.')
            return ""

        return random.choice(candidate_WALKERbp_names)

    def gen_Walker(self, world, num_workers, ref_spawn):
        blueprint_library = world.get_blueprint_library()
        pedestrians = []
        pedestrians_bp = self.choose_walker_name('+workers')
        pedestrians_blueprint = blueprint_library.find(pedestrians_bp)
        max_attempts = 2
        attempts = 0
        for i in range(num_workers):
            spawn_point = self.move_waypoint_forward(ref_spawn, random.randint(0, 5))
            random_yaw = random.uniform(0, 180)
            spawn_npc_point = carla.Transform(
                location=carla.Location(x=spawn_point.transform.location.x, y=spawn_point.transform.location.y,
                                        z=spawn_point.transform.location.z + 0.5),
                rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                        yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                        roll=spawn_point.transform.rotation.roll))
            while attempts < max_attempts:
                try:
                    # 如果位置安全，尝试生成行人
                    npc = world.spawn_actor(pedestrians_blueprint, spawn_npc_point)
                    pedestrians.append(npc)
                    # 新增：记录生成的actor
                    self.generated_actors.append(npc)
                    break  # 成功生成行人，退出循环
                except RuntimeError as e:
                    # 如果生成失败，打印错误信息并尝试新的位置
                    # print(f"Spawn failed at {spawn_npc_point}: {e}")
                    attempts += 1
                    if attempts >= max_attempts:
                        break
                    spawn_point = self.move_waypoint_forward(ref_spawn, random.randint(3, 6))
                    # 重新计算生成点
                    random_yaw = random.uniform(0, 180)
                    spawn_npc_point = carla.Transform(
                        location=carla.Location(x=spawn_point.transform.location.x,
                                                y=spawn_point.transform.location.y,
                                                z=spawn_point.transform.location.z + 0.5),
                        rotation=carla.Rotation(pitch=spawn_point.transform.rotation.pitch,
                                                yaw=spawn_point.transform.rotation.yaw + random_yaw,
                                                roll=spawn_point.transform.rotation.roll))

    def ahead_obstacle_scenario_first(self, world, wp, scene_cfg={}, gen_cfg={}):
        # Desc: 在当前车道的前方生成施工现场
        # 从场景配置字典中获取锥筒数量和间隔距离
        num_cones = scene_cfg.get('num_cones', 5)  # 默认值为5
        cone_interval = scene_cfg.get('cone_interval', 3)  # 默认值为5米
        num_garbage = scene_cfg.get('num_garbage', 50)
        num_workers = scene_cfg.get('num_workers', 4)
        # 1.生成施工牌/水马
        barrier_spawn = self.gen_barrier(world, wp)
        # 2.生成纸板
        ref_spawn = self.gen_creasedbox(world, barrier_spawn)
        # 3.生成锥筒
        last_cone_transform = self.gen_cones(world, barrier_spawn, num_cones, cone_interval)
        # 4.生成垃圾
        self.gen_garbage(world, barrier_spawn, num_garbage)
        # 5.生成行人
        # walker_manager = WalkerManager(world, num_workers, ref_spawn)
        # walker_manager.gen_walkers(num_workers, ref_spawn)
        # self.gen_Walker(world,num_workers, ref_spawn)

        return barrier_spawn.transform.location, barrier_spawn, last_cone_transform

    def ahead_obstacle_scenario_second(self, world, wp, scene_cfg={}, gen_cfg={}):
        # Desc: 在当前车道的前方生成施工现场
        # 从场景配置字典中获取锥筒数量和间隔距离
        num_cones = scene_cfg.get('num_cones', 5)  # 默认值为5
        cone_interval = scene_cfg.get('cone_interval', 3)  # 默认值为5米
        num_workers = scene_cfg.get('num_workers', 4)
        # 1.生成施工牌/水马
        barrier_spawn = self.gen_barrier(world, wp)
        # 2.生成两块钢板
        ref_spawn = self.gen_two_planks(world, barrier_spawn)
        # 3.生成锥筒
        last_cone_transform = self.gen_cones(world, barrier_spawn, num_cones, cone_interval)
        # 4.生成行人
        # walker_manager = WalkerManager(world, num_workers, ref_spawn)
        # walker_manager.gen_walkers(num_workers, ref_spawn)
        # self.gen_Walker(world,num_workers, ref_spawn)
        return barrier_spawn.transform.location, barrier_spawn, last_cone_transform

    def move_waypoint_forward(self, wp, distance):  # 有問題
        # Desc: 将waypoint沿着前进方向移动一定距离
        # dist = 0
        # next_wp = wp
        # while dist < distance:
        #     next_wps = next_wp.next(1)
        #     if not next_wps: # or next_wps[0].is_junction:
        #         break
        #     next_wp = next_wps[0]
        #     dist += 1
        next_wp = wp.next(distance)[0]
        return next_wp

    def gen_two_planks(self, world, barrier_spawn):
        blueprint_library = world.get_blueprint_library()
        plank_bp = self.choose_obsbp_name('+plank')
        print(f"plank : {plank_bp}\n")
        # 计算障碍物的位置
        plank_spawn_first = self.move_waypoint_forward(barrier_spawn, random.randint(2, 3))
        plank_transform_first = plank_spawn_first.transform
        # 从蓝图库中获取障碍物的ActorBlueprint对象
        plank_blueprint_first = blueprint_library.find(plank_bp)
        if plank_blueprint_first is not None:
            new_yaw = plank_transform_first.rotation.yaw - 20
            # 创建一个新的Transform对象，使用新的yaw值
            plank_transform_first = carla.Transform(
                location=carla.Location(x=plank_transform_first.location.x + 0.5,
                                        y=plank_transform_first.location.y - 0.5,
                                        z=plank_transform_first.location.z + 0.5),
                rotation=carla.Rotation(pitch=plank_transform_first.rotation.pitch, yaw=new_yaw,
                                        roll=plank_transform_first.rotation.roll))
            plank_first = world.spawn_actor(plank_blueprint_first, plank_transform_first)
            plank_first.set_simulate_physics(False)  # Ensure the barrier has physics simulation

        plank_spawn_second = self.move_waypoint_forward(barrier_spawn, random.randint(5, 7))
        plank_transform_second = plank_spawn_second.transform
        # 从蓝图库中获取障碍物的ActorBlueprint对象
        plank_blueprint_second = blueprint_library.find(plank_bp)
        if plank_blueprint_second is not None:
            new_yaw_second = plank_transform_second.rotation.yaw - 70
            # 创建一个新的Transform对象，使用新的yaw值
            plank_transform_second = carla.Transform(
                location=carla.Location(x=plank_transform_second.location.x - 0.3,
                                        y=plank_transform_second.location.y + 0.3,
                                        z=plank_transform_second.location.z + 0.5),
                rotation=carla.Rotation(pitch=plank_transform_second.rotation.pitch, yaw=new_yaw_second,
                                        roll=plank_transform_second.rotation.roll))
            plank_second = world.spawn_actor(plank_blueprint_second, plank_transform_second)
            plank_second.set_simulate_physics(False)  # Ensure the barrier has physics simulation
        return plank_spawn_first

    def gen_barrier(self, world, wp):
        blueprint_library = world.get_blueprint_library()
        # 根据过滤条件选择障碍物蓝图
        barrier_bp = self.choose_obsbp_name('+street_barrier')
        print(f"barrier_bp : {barrier_bp}\n")
        # 计算障碍物的位置
        barrier_spawn = self.move_waypoint_forward(wp, random.randint(15, 20))  # debug出來 兩個wp 一樣
        print("barrier_spawn.lane_id", barrier_spawn.lane_id)
        # barrier_spawn = wp.next(random.randint(15, 20))
        print("barrier_spawn", barrier_spawn)
        print("barrier_ego_wp", wp)

        barrier_transform = barrier_spawn.transform
        # 从蓝图库中获取障碍物的ActorBlueprint对象
        barrier_blueprint = blueprint_library.find(barrier_bp)
        if barrier_blueprint is not None:
            new_yaw = barrier_transform.rotation.yaw + 90
            # 创建一个新的Transform对象，使用新的yaw值
            new_transform = carla.Transform(
                location=barrier_transform.location,  # 保持位置不变
                rotation=carla.Rotation(pitch=barrier_transform.rotation.pitch, yaw=new_yaw,
                                        roll=barrier_transform.rotation.roll))
            barrier = world.spawn_actor(barrier_blueprint, new_transform)
            barrier.set_simulate_physics(False)  # Ensure the barrier has physics simulation
            # 新增：记录生成的障碍物
            self.generated_actors.append(barrier)
        return barrier_spawn

    def choose_obsbp_name(self, filters):
        """
        Desc: 根据障碍物类型选择对应的blueprint
        @param filters: +x: 添加类型 -x: 排除类型，按顺序计算
        """
        # garbage: 道路垃圾，废弃物
        # bus_stop: 公交车站
        # construction： 施工
        # street_barrier: 道路指引
        # traffic_barrier: 交通障碍物

        filters = [item.strip() for item in re.split(r'([+\-])', filters.strip()) if item.strip()]

        # 不能为单数
        if len(filters) % 2 != 0:
            return ""

        candidate_obsbp_names = []
        for index in range(0, len(filters), 2):
            op = filters[index]
            filter_type = filters[index + 1]
            if op == '+':
                candidate_obsbp_names.extend(TYPE_OBSTACLE_DICT[filter_type])
            elif op == '-':
                candidate_obsbp_names = list(set(candidate_obsbp_names) - set(TYPE_OBSTACLE_DICT[filter_type]))
            else:
                print(f'Error: {op} is not supported in blueprint choosing.')
                return ""

        if len(candidate_obsbp_names) == 0:
            print(f'Error: candidate_bp_names is empty.')
            return ""

        return random.choice(candidate_obsbp_names)

    def gen_creasedbox(self, world, barrier_spawn):
        blueprint_library = world.get_blueprint_library()
        creasedbox_bp = self.choose_obsbp_name('+creasedbox')
        print(f"creasedbox : {creasedbox_bp}\n")
        # 计算障碍物的位置
        creasedbox_spawn = self.move_waypoint_forward(barrier_spawn, random.randint(15, 20))
        creasedbox_transform = creasedbox_spawn.transform
        # 从蓝图库中获取障碍物的ActorBlueprint对象
        creasedbox_blueprint = blueprint_library.find(creasedbox_bp)
        if creasedbox_blueprint is not None:
            new_yaw = creasedbox_transform.rotation.yaw + 10  # 45
            # 创建一个新的Transform对象，使用新的yaw值
            new_creasedbox_transform = carla.Transform(
                location=carla.Location(x=creasedbox_transform.location.x, y=creasedbox_transform.location.y,
                                        z=creasedbox_transform.location.z + 0.5),
                rotation=carla.Rotation(pitch=creasedbox_transform.rotation.pitch, yaw=new_yaw,
                                        roll=creasedbox_transform.rotation.roll))
            creasedbox = world.spawn_actor(creasedbox_blueprint, new_creasedbox_transform)
            creasedbox.set_simulate_physics(False)  # Ensure the barrier has physics simulation
            # 新增：记录生成的对象
            self.generated_actors.append(creasedbox)
        return creasedbox_spawn

    def gen_garbage(self, world, barrier_spawn, num_garbage):
        blueprint_library = world.get_blueprint_library()
        gar_spawn = self.move_waypoint_forward(barrier_spawn, random.randint(2, 3))
        for i in range(num_garbage):
            garbage_bp = self.choose_obsbp_name('+garbage')
            garbage_blueprint = blueprint_library.find(garbage_bp)
            k_gar_soffset = random.choice([2, 3])
            x_gar_soffset = random.uniform(0, 0.3)
            y_gar_soffset = random.uniform(0, 0.5)
            yaw_gar_soffset = random.uniform(0, 360)
            spawn_garbage_point = carla.Transform(
                location=carla.Location(
                    x=gar_spawn.transform.location.x + 1.0 + (-1) ** (-1 * k_gar_soffset) * x_gar_soffset * 0.4,
                    y=gar_spawn.transform.location.y + (-1) ** (k_gar_soffset) * y_gar_soffset * 0.3,
                    z=gar_spawn.transform.location.z + 0.5),
                rotation=carla.Rotation(pitch=gar_spawn.transform.rotation.pitch,
                                        yaw=gar_spawn.transform.rotation.yaw + (-1) ** (
                                            k_gar_soffset) * yaw_gar_soffset,
                                        roll=gar_spawn.transform.rotation.roll)
            )
            while True:
                garbage = world.spawn_actor(garbage_blueprint, spawn_garbage_point)
                garbage.set_simulate_physics(False)
                # 新增：记录生成的对象
                self.generated_actors.append(garbage)
                break
        return gar_spawn

    def gen_cones(self, world, barrier_spawn, num_cones, cone_interval):
        blueprint_library = world.get_blueprint_library()
        cone_bp = self.choose_obsbp_name('+traffic_barrier')
        cone_blueprint = blueprint_library.find(cone_bp)
        if cone_bp is None:
            raise ValueError("Traffic cone blueprint not found in the library.")

        # Get the waypoint just ahead of the barrier
        _map = world.get_map()
        barrier_waypoint = _map.get_waypoint(barrier_spawn.transform.location)
        first_cone_waypoint = barrier_waypoint.next(0.3)[0]  # Get the next waypoint after the barrier

        # Spawn the traffic cones
        for i in range(num_cones):
            try:
                # 尝试获取锥筒的目标waypoint
                target_waypoint = first_cone_waypoint.next((i + 1) * int(cone_interval))[0]
                if target_waypoint is not None:  # 确保waypoint是有效的
                    assert isinstance(target_waypoint, carla.Waypoint)
                    # 计算锥筒的位置
                    cone_left_location = carla.Location(
                        x=target_waypoint.transform.location.x + (
                                ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        y=target_waypoint.transform.location.y - (
                                ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        z=target_waypoint.transform.location.z)

                    # 创建锥筒的变换对象
                    cone_left_transform = carla.Transform(
                        location=cone_left_location,
                        rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                                yaw=target_waypoint.transform.rotation.yaw,
                                                roll=target_waypoint.transform.rotation.roll))
                    # 在计算出的位置和方向上生成锥筒
                    cone = world.spawn_actor(cone_blueprint, cone_left_transform)
                    cone.set_simulate_physics(False)
                    # 新增：记录生成的对象
                    self.generated_actors.append(cone)

                    cone_right_location = carla.Location(
                        x=target_waypoint.transform.location.x - (
                                ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        y=target_waypoint.transform.location.y + (
                                ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                            math.radians(target_waypoint.transform.rotation.yaw))),
                        z=target_waypoint.transform.location.z)

                    # 创建锥筒的变换对象
                    cone_right_transform = carla.Transform(
                        location=cone_right_location,
                        rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                                yaw=target_waypoint.transform.rotation.yaw,
                                                roll=target_waypoint.transform.rotation.roll))
                    # 在计算出的位置和方向上生成锥筒
                    cone = world.spawn_actor(cone_blueprint, cone_right_transform)
                    cone.set_simulate_physics(False)
                    # 新增：记录生成的对象
                    self.generated_actors.append(cone)

                else:
                    print(f"Invalid waypoint for cone placement at i = {i}")
            except RuntimeError as e:
                print(f"Error placing cones at i = {i}: {e}")

            for i in range(num_cones):
                try:
                    # 尝试获取锥筒的目标waypoint
                    target_waypoint = first_cone_waypoint.next((i + 1) * int(cone_interval))[0]
                    if target_waypoint is not None:  # 确保waypoint是有效的
                        assert isinstance(target_waypoint, carla.Waypoint)
                        # 计算锥筒的位置
                        cone_left_location = carla.Location(
                            x=target_waypoint.transform.location.x + (
                                    ((target_waypoint.lane_width - 0.8) / 2) * math.sin(
                                math.radians(target_waypoint.transform.rotation.yaw))),
                            y=target_waypoint.transform.location.y - (
                                    ((target_waypoint.lane_width - 0.8) / 2) * math.cos(
                                math.radians(target_waypoint.transform.rotation.yaw))),
                            z=target_waypoint.transform.location.z)

                        # 创建锥筒的变换对象
                        cone_left_transform = carla.Transform(
                            location=cone_left_location,
                            rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                                    yaw=target_waypoint.transform.rotation.yaw,
                                                    roll=target_waypoint.transform.rotation.roll))
                        # 在计算出的位置和方向上生成锥筒
                        cone = world.spawn_actor(cone_blueprint, cone_left_transform)
                        cone.set_simulate_physics(False)
                        # 新增：记录生成的对象
                        self.generated_actors.append(cone)

                        cone_right_location = carla.Location(
                            x=target_waypoint.transform.location.x - (
                                    ((target_waypoint.lane_width - 1) / 2) * math.sin(
                                math.radians(target_waypoint.transform.rotation.yaw))),
                            y=target_waypoint.transform.location.y + (
                                    ((target_waypoint.lane_width - 1) / 2) * math.cos(
                                math.radians(target_waypoint.transform.rotation.yaw))),
                            z=target_waypoint.transform.location.z)

                        # 创建锥筒的变换对象
                        cone_right_transform = carla.Transform(
                            location=cone_right_location,
                            rotation=carla.Rotation(pitch=target_waypoint.transform.rotation.pitch,
                                                    yaw=target_waypoint.transform.rotation.yaw,
                                                    roll=target_waypoint.transform.rotation.roll))
                        # 在计算出的位置和方向上生成锥筒
                        cone = world.spawn_actor(cone_blueprint, cone_right_transform)
                        cone.set_simulate_physics(False)
                        # 新增：记录生成的对象
                        self.generated_actors.append(cone)

                        # 如果是最后一个锥筒，存储其位置
                        if i == num_cones - 1:
                            last_cone_transform = cone_left_transform
                            print('-------last_cone_transform', last_cone_transform)
                            # Spawn the last barrier
                            last_barrier_bp = self.choose_obsbp_name('+street_barrier')
                            print(f"last_barrier_bp : {last_barrier_bp}\n")
                            # 从蓝图库中获取障碍物的ActorBlueprint对象
                            last_barrier_blueprint = blueprint_library.find(last_barrier_bp)
                            if last_barrier_blueprint is not None:
                                new_transform_last = carla.Transform(
                                    location=carla.Location(x=last_cone_transform.location.x - (
                                            target_waypoint.lane_width - 0.8) / 2 * math.sin(
                                        math.radians(last_cone_transform.rotation.yaw)),
                                                            y=last_cone_transform.location.y + (
                                                                    target_waypoint.lane_width - 0.8) / 2 * math.cos(
                                                                math.radians(last_cone_transform.rotation.yaw)),
                                                            z=last_cone_transform.location.z + 0.1),  # 保持位置不变
                                    rotation=carla.Rotation(pitch=last_cone_transform.rotation.pitch,
                                                            yaw=last_cone_transform.rotation.yaw - 90,
                                                            roll=last_cone_transform.rotation.roll))
                                last_barrier = world.spawn_actor(last_barrier_blueprint, new_transform_last)
                                print("-------", last_barrier)
                                last_barrier.set_simulate_physics(False)  # Ensure the barrier has physics simulation
                                # 新增：记录生成的对象
                                self.generated_actors.append(last_barrier)
                                return new_transform_last
                    else:
                        print(f"Invalid waypoint for cone placement at i = {i}")
                except RuntimeError as e:
                    print(f"Error placing cones at i = {i}: {e}")



    def random_spawn_point(self, world_map: carla.Map, different_from: carla.Location = None) -> carla.Transform:
        available_spawn_points = world_map.get_spawn_points()

        if different_from is not None:
            while True:
                # print("！！！！！！！！！！！！！！")
                current_waypoint = world_map.get_waypoint(different_from.location)
                spawn_point = random.choice(current_waypoint.next(15.0))  # 吧这里的available_spawn_points改成
                spawn_point_location = spawn_point.transform.location
                if spawn_point_location != different_from:
                    return spawn_point.transform
        else:
            return random.choice(available_spawn_points)

    def choose_points(self, map):
        # 生成起终点
        self.origin_point, self.selected_task = self.choose_tasks()
        # debug
        x = self.origin_point['x']
        y = self.origin_point['y']
        z = self.origin_point['z']
        yaw = self.origin_point['yaw']
        # debug
        # x = 55.587372
        # y=-203.109268
        # yaw=-178.560471
        # z=0.27

        print('generate point')

        self.origin_location = carla.Location(x=x, y=y, z=z)
        print("!!!!!origin_location!!!!!!", self.origin_location)
        self.origin = carla.Transform(location=carla.Location(x=x, y=y, z=z),
                                      rotation=carla.Rotation(pitch=0, yaw=yaw, roll=0))

        # choose destination (final point)
        self.destination_location = self.random_spawn_point(map, different_from=self.origin).location

        print("!!!!!destination_location!!!!!!", self.destination_location)

        return self.origin_location, self.destination_location

    def get_next_waypoints(self, map, ego, amount: int) -> list:
        """
        使用 CARLA API 获取自车前方指定数量的路径点。

        Args:
            amount (int): 需要获取的路径点数量。

        Returns:
            list: 一个包含 carla.Waypoint 对象的列表，表示自车前方的路径点。
                  如果无法获取足够数量的路径点，将返回尽可能多的路径点。
        """
        current_waypoint = map.get_waypoint(ego.get_location())

        next_waypoints = []
        if current_waypoint is None:
            print("警告: 无法获取自车当前所在的 Waypoint。")
            return []

        current_search_wp = current_waypoint
        for _ in range(amount):
            next_wps_list = current_search_wp.next(1.0)  # 沿着车道向前 1 米

            if not next_wps_list:
                break

            next_wp = next_wps_list[0]
            next_waypoints.append(next_wp)

            current_search_wp = next_wp

        return next_waypoints

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

    def _extract_map_elements(self, ego_transform, map):
        """辅助函数：提取车道线和道路边界"""
        elements = []
        waypoint = map.get_waypoint(ego_transform.location)

        for i in range(-2, 3):  # 检查当前、左二、右二共5条车道
            wp = waypoint
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                continue
            # if i < 0:
            #     for _ in range(abs(i)): wp = wp.get_left_lane()
            # elif i > 0:
            #     for _ in range(i): wp = wp.get_right_lane()
            if i < 0:
                for _ in range(abs(i)):
                    if wp is None:  # 在尝试获取前检查
                        break  # 如果已经为 None，则不再继续获取
                    wp = wp.get_left_lane()
            elif i > 0:
                for _ in range(i):
                    if wp is None:  # 在尝试获取前检查
                        break  # 如果已经为 None，则不再继续获取
                    wp = wp.get_right_lane()
            if wp is not None:

                # 提取车道中心线作为 'lane_divider'
                wps_center = wp.next_until_lane_end(20.0)  # 向前追溯20米
                if len(wps_center) > 1:
                    points_center = self._wps_to_ego_points(wps_center, ego_transform)
                    elements.append(('lane_divider', points_center))

                # 提取道路边界
                wps_boundary = wp.next_until_lane_end(20.0)
                if len(wps_boundary) > 1:
                    points_boundary = self._wps_to_ego_points(wps_boundary, ego_transform, boundary='right')  # 取右边界
                    elements.append(('road_boundary', points_boundary))

        return elements

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

    def _get_vector_map_features(self, ego_transform, world, map):
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
        ego_matrix = np.array(ego_transform.get_matrix())

        # --- 2. 填充动态物体信息区 (Dynamic Objects Block) ---

        # 获取所有车辆和行人actor
        actors = world.get_actors().filter('vehicle.*|walker.pedestrian.*')

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
        map_elements = self._extract_map_elements(ego_transform, map)

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

    def save_batch_data(self, force_save=False):
        """批量保存数据并清理内存"""
        global dataset_records

        if len(dataset_records) >= SAVE_BATCH_SIZE or force_save:
            if len(dataset_records) > 0:
                SAVE_DIR = "expert_data_high"
                os.makedirs(SAVE_DIR, exist_ok=True)

                # 生成时间戳
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                # 根据交通密度生成文件名前缀
                if self.traffic_density == "low":
                    filename_prefix = "simple_scenario"
                elif self.traffic_density == "medium":
                    filename_prefix = "medium_scenario"
                elif self.traffic_density == "high":
                    filename_prefix = "complex_scenario"
                else:
                    filename_prefix = "unknown_scenario"

                pickle_file_path = os.path.join(SAVE_DIR, f"{filename_prefix}_batch_data_{timestamp}.pkl")

                # 保存当前批次数据
                try:
                    with open(pickle_file_path, 'wb') as f:
                        pickle.dump(dataset_records, f)
                    print(f"批量保存了 {len(dataset_records)} 条记录到 {pickle_file_path}")

                    # 清理内存中的数据
                    dataset_records.clear()
                    print("内存中的数据已清理")
                except Exception as e:
                    print(f"保存数据时发生错误: {e}")

    def cleanup_actors(self):
        """清理生成的CARLA对象"""
        cleanup_count = 0
        for actor in self.generated_actors:
            try:
                if actor.is_alive:
                    actor.destroy()
                    cleanup_count += 1
            except:
                pass  # 忽略已经销毁的对象

        self.generated_actors.clear()
        if cleanup_count > 0:
            print(f"清理了 {cleanup_count} 个CARLA对象")

    def record_data(self, ego, map, world, ego_transform):
        # 1. 记录 Vehicle 信息 (throttle, steer, brake)
        control = ego.get_control()
        ego_speed_x = ego.get_velocity().x
        ego_speed_y = ego.get_velocity().y
        ego_speed_z = ego.get_velocity().z
        ego_acc = ego.get_acceleration()
        vehicle_data = {
            'throttle': control.throttle,
            'steer': control.steer,
            'brake': control.brake,
            'acc_x': ego_acc.x,
            'acc_y': ego_acc.y,
            'acc_z': ego_acc.z,
            'speed_x': ego_speed_x,
            'speed_y': ego_speed_y,
            'speed_z': ego_speed_z
        }

        # 2. 记录 Navigation 信息 (当前路径点，也可以是未来几个路径点)
        # 这里记录当前自车所在的车道中心点的xy坐标作为导航信息
        navigation_data = self._get_navigation_features(ego, map)

        # 3. 记录 Vector Map 信息
        vector_map_features = self._get_vector_map_features(ego_transform, world, map)
        # 由于 vector_map_features 是一个 numpy 数组，为了方便存储，可以将其转换为列表
        vector_map_data = vector_map_features.tolist()

        # 将所有数据打包到一个字典中，并添加到数据集列表中
        current_record = {
            'timestamp': time.time(),  # 添加时间戳
            'vehicle_info': vehicle_data,
            'navigation_info': navigation_data,
            'vector_map': vector_map_data,
            'ego_stage': None,  # 初始为None，后面会更新
            'reason': None,  # 初始为None，后面会更新
            'distance_to_obstacle': None,  # 初始为None，后面会更新
        }

        dataset_records.append(current_record)

        # 新增：检查是否需要批量保存
        self.save_batch_data()

        return current_record

    def _tf_set_ego_route(self, ego, route, traffic_manager):
        if self._check_tf(traffic_manager):
            traffic_manager.set_route(ego, route)

    def _tf_disable_ego_auto_lane_change(self, traffic_manager, ego):
        if self._check_tf(traffic_manager):
            traffic_manager.auto_lane_change(ego, False)

    def _set_ego_autopilot(self, ego):
        ego.set_autopilot(enabled=True)

    def _check_tf(self, traffic_manager):
        if traffic_manager is None:
            print(f'Please set traffic manager by scenario.set_tf(tf_instance) first!')
            return False
        return True

    def calc_relative_position(self, ego, actor, only_azimuth=False, ego_speed=-1.0, actor_speed=-1.0):
        # Desc: 计算相对位置
        if isinstance(ego, carla.Transform):
            ego_transform = ego
        elif isinstance(ego, carla.Actor):
            ego_transform = ego.get_transform()
        else:
            raise NotImplementedError

        if isinstance(actor, carla.Transform):
            actor_transform = actor
        elif isinstance(actor, carla.Actor):
            actor_transform = actor.get_transform()
        else:
            raise NotImplementedError

        # Desc: 计算他车相对于自车的方位角
        # Special: 自车前进向量顺时针旋转到自车指向他车的向量的角度（360度制）
        v1 = ego_transform.get_forward_vector()
        v2 = actor_transform.location - ego_transform.location
        v1 = np.array([-v1.x, v1.y])
        v2 = np.array([-v2.x, v2.y])
        v2 = normalize(v2)
        v12_cos_value = np.dot(v1, v2)
        v12_cos_value = np.clip(v12_cos_value, -1, 1)
        v12_sin_value = np.cross(v1, v2)
        v12_sin_value = np.clip(v12_sin_value, -1, 1)
        v12_180_angle = math.degrees(math.acos(v12_cos_value))
        v12_360_angle = 360 - v12_180_angle if v12_sin_value > 0 else v12_180_angle

        if only_azimuth:
            return v12_360_angle

        # Desc: 计算他车相对于自车是驶来还是驶去还是静止
        # Special: 驶来条件1：自车前进向量与自车指向他车向量的180度制角度小于90度
        #          驶来条件2：自车前进向量与他车前进向量180度制角度大于90度
        #          驶来条件3：自车速度大于他车速度
        if ego_speed == -1.0:
            ego_speed = math.sqrt(ego.get_velocity().x ** 2 + ego.get_velocity().y ** 2) * 3.6
        if actor_speed == -1.0:
            actor_speed = math.sqrt(actor.get_velocity().x ** 2 + actor.get_velocity().y ** 2) * 3.6

        if actor_speed > 0.1:
            v3 = actor_transform.get_forward_vector()
            v3 = np.array([-v3.x, v3.y])

            ego_loc = np.array([-ego_transform.location.x, ego_transform.location.y])
            actor_loc = np.array([-actor_transform.location.x, actor_transform.location.y])

            old_distance = math.sqrt((ego_loc[0] - actor_loc[0]) ** 2 + (ego_loc[1] - actor_loc[1]) ** 2)
            actor_next_loc = actor_loc + v3 * actor_speed / 3.6 / 20
            ego_next_loc = ego_loc + v1 * ego_speed / 3.6 / 20
            next_distance = math.sqrt(
                (ego_next_loc[0] - actor_next_loc[0]) ** 2 + (ego_next_loc[1] - actor_next_loc[1]) ** 2)

            if abs(next_distance - old_distance) > 0.1 or True:
                if next_distance < old_distance:
                    relative_direction = 'approaching'
                else:
                    relative_direction = 'leaving'
            else:
                relative_direction = ''
        else:
            relative_direction = 'stilling'

        return v12_360_angle, relative_direction

