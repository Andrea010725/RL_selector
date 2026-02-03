# -*- coding: utf-8 -*-
# @Time    : 2023/11/4 下午8:35
# @Author  : Hcyang
# @File    : tiny_scenarios.py
# @Desc    : xxx

import re
import random
import carla
import numpy as np
from random import choice
import os
import argparse
import pickle
import json
# import ipdb


# Effect: 每个scenario的返回值为：[(bp_name, transform), ...]  e.g. [('vehicle.tesla.model3', transform), ...]


VEHICLE_TYPE_DICT = {
    'vehicle.audi.a2': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.audi.etron': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.audi.tt': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.bmw.grandtourer': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.chevrolet.impala': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.citroen.c3': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.dodge.charger_2020': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.dodge.charger_police': ['car', 'special', 'police', 'wheel4'],
    'vehicle.dodge.charger_police_2020': ['car', 'special', 'police', 'wheel4'],
    'vehicle.ford.crown': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.ford.mustang': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.jeep.wrangler_rubicon': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.lincoln.mkz_2017': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.lincoln.mkz_2020': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.mercedes.coupe': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.mercedes.coupe_2020': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.micro.microlino': ['car', 'small', 'wheel4'],
    'vehicle.mini.cooper_s': ['car', 'wheel4', 'common'],
    'vehicle.mini.cooper_s_2021': ['car', 'wheel4', 'common'],
    'vehicle.nissan.micra': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.nissan.patrol': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.nissan.patrol_2021': ['car', 'suv', 'wheel4', 'common', 'hcy1'],
    'vehicle.seat.leon': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.tesla.model3': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.toyota.prius': ['car', 'wheel4', 'common', 'hcy1'],
    'vehicle.carlamotors.carlacola': ['truck', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.carlamotors.firetruck': ['truck', 'special', 'fire', 'large', 'wheel4'],
    'vehicle.tesla.cybertruck': ['truck', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.ford.ambulance': ['van', 'special', 'ambulance', 'large', 'wheel4'],
    'vehicle.mercedes.sprinter': ['van', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.volkswagen.t2': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.volkswagen.t2_2021': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.mitsubishi.fusorosa': ['bus', 'large', 'wheel4', 'common', 'hcy1'],
    'vehicle.harley-davidson.low_rider': ['moto', 'wheel2', 'common'],
    'vehicle.kawasaki.ninja': ['moto', 'wheel2', 'common'],
    'vehicle.vespa.zx125': ['electric', 'wheel2'],
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


def choose_bp_name(filters):
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
    # hcy1: huchuanyang自定义的车辆集合

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


def _vehicle_flow_scenario(wp, direction='left', filters='+common', idp=0.8):
    # Desc: 在当前waypoint的左侧车道或者右侧车道生成车流
    if direction == 'right':
        other_lane = wp.get_right_lane()
    elif direction == 'left':
        other_lane = wp.get_left_lane()
    else:
        raise NotImplementedError
    results = []

    # Desc: 先向前生成车流
    _vehicle_wp = other_lane
    right_forward_index = 1
    while right_forward_index <= random.randint(4, 8):   # 4  6
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            results.append((bp_name, _vehicle_wp.transform))
        _vehicle_wp = _vehicle_wp.next(random.randint(8, 15))[0]
        right_forward_index += 1

    # Desc: 再向后生成车流
    _vehicle_wp = other_lane
    right_backward_index = 1
    while right_backward_index <= random.randint(3, 7):   # 1 4
        _vehicle_wp = _vehicle_wp.previous(8)[0]
        bp_name = choose_bp_name(filters)
        if random.random() < idp:
            results.append((bp_name, _vehicle_wp.transform))
        right_backward_index += 1

    return results


def right_lane_vehicle_flow_scenario(wp, filters='+common'):
    # Desc: 在当前车道的右侧车道生成车流
    return _vehicle_flow_scenario(wp, direction='right', filters=filters)


def left_lane_vehicle_flow_scenario(wp, filters='+common'):
    # Desc: 在当前车道的左侧车道生成车流
    return _vehicle_flow_scenario(wp, direction='left', filters=filters)


def box_obstacle_one_lane_scenario(wp):
    # Desc: 在当前waypoint的位置生成多个box障碍物阻挡当前车道
    box_bps = ['static.prop.box01', 'static.prop.box02', 'static.prop.box03']
    box_num = random.randint(1, 3)

    unit = wp.lane_width / 4.0
    ori_x = wp.transform.location.x
    ori_y = wp.transform.location.y
    possible_box_locs = []
    for x_offset in [unit, 0, -unit]:
        for y_offset in [unit, 0, -unit]:
            possible_box_locs.append(carla.Location(x=ori_x+x_offset, y=ori_y+y_offset, z=wp.transform.location.z + 1.0))

    possible_box_rots = []
    for _ in range(50):
        pitch_offset = random.randint(0, 360)
        yaw_offset = random.randint(0, 360)
        roll_offset = random.randint(0, 360)
        possible_box_rots.append(carla.Rotation(pitch=pitch_offset, yaw=yaw_offset, roll=roll_offset))

    results = []
    for _ in range(box_num):
        disturb_loc = gen_location_disturbance(scale=unit / 4.0)
        box_loc = random.choice(possible_box_locs) + disturb_loc
        box_rot = random.choice(possible_box_rots)
        box_bp = random.choice(box_bps)
        box_transform = carla.Transform(box_loc, box_rot)
        results.append((box_bp, box_transform))
    print(f'box obstacle num: {box_num}')

    return results


def cone_obstacle_one_lane_scenario(wp):
    # Desc: 在当前waypoint的位置生成多个box障碍物阻挡当前车道
    cone_bp = 'static.prop.constructioncone'

    wp_loc_z = wp.transform.location.z
    unit = wp.lane_width / 8.0 * 3.0
    right_vec = wp.transform.get_right_vector()
    left_vec = right_vec * -1.0

    pre_wp = wp.previous(1)[0]
    next_wp = wp.next(1)[0]

    results = []
    for cur_wp in [pre_wp, next_wp]:
        results.append((cone_bp, cur_wp.transform))
        tmp = cur_wp.transform.location + right_vec * unit
        results.append((cone_bp, carla.Transform(carla.Location(x=tmp.x, y=tmp.y, z=wp_loc_z))))
        tmp = cur_wp.transform.location + left_vec * unit
        results.append((cone_bp, carla.Transform(carla.Location(x=tmp.x, y=tmp.y, z=wp_loc_z))))

    cur_wp = wp
    center_bp = choice(['static.prop.warningconstruction'])
    results.append((center_bp, carla.Transform(wp.transform.location, carla.Rotation(yaw=wp.transform.rotation.yaw+90))))
    tmp = cur_wp.transform.location + right_vec * unit
    results.append((cone_bp, carla.Transform(carla.Location(x=tmp.x, y=tmp.y, z=wp_loc_z))))
    tmp = cur_wp.transform.location + left_vec * unit
    results.append((cone_bp, carla.Transform(carla.Location(x=tmp.x, y=tmp.y, z=wp_loc_z))))

    return results


def barrier_obstacle_one_lane_scenario(wp):
    # Desc: 在当前waypoint的位置生成多个box障碍物阻挡当前车道
    barrier_bp = 'static.prop.streetbarrier'

    wp_loc_z = wp.transform.location.z
    unit = wp.lane_width / 8.0 * 3.0
    right_vec = wp.transform.get_right_vector()
    left_vec = right_vec * -1.0

    next_wp = wp.next(2)[0]

    results = []
    cur_wp = wp
    results.append((barrier_bp, carla.Transform(wp.transform.location, carla.Rotation(yaw=wp.transform.rotation.yaw+90))))
    cur_wp = next_wp
    results.append((barrier_bp, carla.Transform(wp.transform.location, carla.Rotation(yaw=wp.transform.rotation.yaw+90))))

    return results


def gen_location_disturbance(scale):
    # Desc: 生成一个随机的扰动Location
    # Param: scale: 扰动的大小
    # Return: carla.Location
    x_offset = random.uniform(-scale, scale)
    y_offset = random.uniform(-scale, scale)
    return carla.Location(x=x_offset, y=y_offset, z=0)