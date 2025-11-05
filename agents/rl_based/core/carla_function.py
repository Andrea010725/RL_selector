"""
部分carla函数
"""
import argparse
import math
import os
import carla
import time
from time import sleep

import ipdb


class VehicleFunction:
    def __init__(self, world, hero_vehicle):                    #, world, hero_vehicle
        self._world = world
        # self.last_active_time = {}
        self.hero_vehicle = hero_vehicle

    def get_vehicle_location(self, vehicle):
        location = vehicle.get_location()
        return location

    def get_vehicle_velocity(self, vehicle):
        velocity = vehicle.get_velocity()
        return velocity

    # add function 补充函数
    def get_vehicle_id(self, vehicle):
        #   补充代码  需要获取actor的id type等
        # id = vehicle.get_actor()
        # 函数待补充 主要作用是 分类
        return id


    # 判断是否在自车前面  （带朝向的）
    def is_in_front_of_hero_vehicle(self, other_vehicle):
        hero_location = self.hero_vehicle.get_location()
        hero_rotation = self.hero_vehicle.get_transform().rotation
        other_location = other_vehicle.get_location()
        relative_position = carla.Vector3D(
            other_location.x - hero_location.x,
            other_location.y - hero_location.y,
            other_location.z - hero_location.z
        )
        hero_front_vector = hero_rotation.get_forward_vector()
        k = 0.8
        l = 10
        if other_location.x <= hero_location.x + l * math.cos(hero_rotation.yaw):
            if other_location.y <= hero_location.y + l * math.sin(hero_rotation.yaw):
                return True
        return False


    def check_surrounding_vehicles(self, hero_vehicle):
        surrounding_info = []
        surrounding_vehicles = []
        N = 0
        # while True:
        if hero_vehicle is None:
            print("can not find ego_vehicle!!!")
        else:
            # print("find ego_vehicle!!!")
            hero_location = hero_vehicle.get_location()
            for vehicle in self._world.get_actors().filter('vehicle.*'):
                surrounding_vehicles.append(vehicle)
                if N == 0:
                    # print(f"there are  { len(self._world.get_actors().filter('vehicle.*'))} Vehicles in the Town!!!")
                    N = N + 1

                if vehicle.id == hero_vehicle.id:
                    continue

                else:
                    # ipdb.set_trace()
                    vehicle_location = self.get_vehicle_location(vehicle)
                    vehicle_velocity = self.get_vehicle_velocity(vehicle)
                    distance1 = math.sqrt(
                        (hero_location.x - vehicle_location.x) ** 2 +
                        (hero_location.y - vehicle_location.y) ** 2
                    )
                    if distance1 <= 20:  # 距离20米之内
                        if self.is_in_front_of_hero_vehicle(vehicle):
                            # print("detecting One Vehicle satisfying the in_front_of_hero_vehicle function!!!")
                            # print("!!!!!!  Other_Vehicle_velocity", vehicle_velocity)
                            surrounding_info.append({
                                'location': {
                                    'x': vehicle_location.x,
                                    'y': vehicle_location.y,
                                    'z': vehicle_location.z
                                },
                                'velocity': {
                                    'x': vehicle_velocity.x,
                                    'y': vehicle_velocity.y,
                                    'z': vehicle_velocity.z
                                }
                            })
        # print("Finished checking surrounding vehicles")
        # print("surrounding_info", surrounding_info)
        # print("surrounding_vehicles", surrounding_vehicles)
        return surrounding_info, surrounding_vehicles