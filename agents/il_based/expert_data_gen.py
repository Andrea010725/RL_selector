from __future__ import annotations
import math
import random
import sys
from types import SimpleNamespace
# from scipy.optimize import minimize # 这个库在当前代码中未使用，可以注释掉
import pandas as pd  # 用于保存CSV
import datetime  # 用于生成带时间戳的文件名

from typing import List, Tuple

# 请确保您的CARLA egg路径正确
sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np
import ipdb

# 你的工程内模块
# 注意：确保这里的路径是您本地RL_selector项目的根路径
sys.path.append("/home/ajifang/czw/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.tools import SceneManager

import os
import json


class ImitationDataCollector:
    """
    一个用于模仿学习的数据采集器。
    - 增加了碰撞检测功能，发生碰撞后会停止采集。
    """

    def __init__(self, client: carla.Client, ego_vehicle: carla.Actor, output_dir: str = "./"):
        self.client = client
        self.world = self.client.get_world()
        self.ego = ego_vehicle

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.main_output_dir = os.path.join(output_dir, f"expert_data_{timestamp}")
        self.images_dir = os.path.join(self.main_output_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        print(f"[Collector] 数据将保存至: {self.main_output_dir}")

        self.metadata_path = os.path.join(self.main_output_dir, "metadata.csv")
        self.records = []
        self.frame_counter = 0

        # --- 新增 ---
        # 用于碰撞检测的标志位
        self.has_collided = False

        # 初始化传感器
        self.camera_rgb = None
        self.collision_sensor = None  # <--- 新增
        self.latest_image = None
        self._setup_sensors()

    def _setup_sensors(self):
        """创建并设置前视摄像头和碰撞传感器"""
        bp_library = self.world.get_blueprint_library()

        # --- 设置摄像头 (不变) ---
        camera_bp = bp_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '256')
        camera_bp.set_attribute('image_size_y', '256')
        camera_bp.set_attribute('fov', '90')
        transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera_rgb = self.world.spawn_actor(camera_bp, transform, attach_to=self.ego)
        self.camera_rgb.listen(lambda image: self._on_camera_image(image))

        # --- 新增：设置碰撞传感器 ---
        collision_bp = bp_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _on_camera_image(self, image: carla.Image):
        self.latest_image = image

    # --- 新增：碰撞传感器的回调函数 ---
    def _on_collision(self, event: carla.CollisionEvent):
        """当碰撞发生时，此函数会被调用"""
        # 获取与EGO碰撞的另一个actor
        other_actor = event.other_actor
        print(f"[碰撞检测] �� EGO 车辆与 {other_actor.type_id} (ID: {other_actor.id}) 发生碰撞！")

        # 设置碰撞标志位
        self.has_collided = True

    def tick(self, control: carla.VehicleControl):
        # tick 函数内容不变
        if self.latest_image is None:
            return
        image_path_rel = os.path.join("images", f"{self.frame_counter:06d}.png")
        image_path_abs = os.path.join(self.main_output_dir, image_path_rel)
        self.latest_image.save_to_disk(image_path_abs)

        ego_transform = self.ego.get_transform()
        ego_velocity = self.ego.get_velocity()
        ego_speed = ego_velocity.length()

        other_actors_info = []
        all_actors = self.world.get_actors()
        for actor in all_actors:
            if actor.id == self.ego.id or "spectator" in actor.type_id:
                continue
            if 'vehicle' in actor.type_id or 'walker' in actor.type_id or 'traffic_light' in actor.type_id or 'static.prop' in actor.type_id:
                actor_tf = actor.get_transform()
                actor_vel = actor.get_velocity()
                info = {
                    'type': actor.type_id,
                    'location': {'x': actor_tf.location.x, 'y': actor_tf.location.y, 'z': actor_tf.location.z},
                    'rotation': {'pitch': actor_tf.rotation.pitch, 'yaw': actor_tf.rotation.yaw,
                                 'roll': actor_tf.rotation.roll},
                    'velocity': {'x': actor_vel.x, 'y': actor_vel.y, 'z': actor_vel.z},
                    'speed': actor_vel.length(),
                    'bounding_box_extent': {'x': actor.bounding_box.extent.x, 'y': actor.bounding_box.extent.y,
                                            'z': actor.bounding_box.extent.z}
                }
                other_actors_info.append(info)

        record = {
            'frame_id': self.frame_counter,
            'image_path': image_path_rel,
            'ego_speed': ego_speed,
            'ego_throttle': control.throttle,
            'ego_brake': control.brake,
            'ego_loc_x': ego_transform.location.x,
            'ego_loc_y': ego_transform.location.y,
            'ego_loc_z': ego_transform.location.z,
            'other_actors_info': json.dumps(other_actors_info)
        }
        self.records.append(record)
        self.frame_counter += 1

    def save_to_disk(self):
        # save_to_disk 函数内容不变
        if not self.records:
            print("[Collector] 警告: 没有任何数据被记录。")
            return
        df = pd.DataFrame(self.records)
        df.to_csv(self.metadata_path, index=False)
        print(f"[Collector] 成功将 {len(self.records)} 条记录的元数据保存到 {self.metadata_path}")

    def destroy_sensors(self):
        """在程序结束时销毁所有传感器"""
        if self.camera_rgb and self.camera_rgb.is_alive:
            self.camera_rgb.stop()
            self.camera_rgb.destroy()
            print("[Collector] 前视摄像头已销毁。")
        # --- 新增：销毁碰撞传感器 ---
        if self.collision_sensor and self.collision_sensor.is_alive:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            print("[Collector] 碰撞传感器已销毁。")

class ExpertController:
    """
    一个基于规则的专家控制器，用于实现换道规避逻辑。
    """

    def __init__(self, ego_vehicle: carla.Actor, last_cone_transform: carla.Transform, traffic_manager: carla.TrafficManager):
        self.ego = ego_vehicle
        self.world = self.ego.get_world()
        self.amap = self.world.get_map()
        self.last_cone_tf = last_cone_transform

        # 控制器状态机
        self.state = "DRIVING_FORWARD"  # DRIVING_FORWARD, CHANGING_LANE, AUTOPILOT
        self.lane_change_start_wp = None
        self.lane_change_direction = None
        self.lane_change_steps = 0  #

        self.tm = traffic_manager

    def run_step(self):
        """
        执行一个决策步骤。注意：此函数不返回控制指令，而是直接调用TM API。
        """
        # 如果任务已完成，则不执行任何操作

        if self.state == "MANEUVER_COMPLETE":
            return

        ego_tf = self.ego.get_transform()
        ego_wp = self.amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        # 状态1: 车辆在自动驾驶下正常巡航，等待时机
        if self.state == "DRIVING_FORWARD":
            dist_to_cone = ego_tf.location.distance(self.last_cone_tf.location)

            if dist_to_cone < 13.0:
                print("[Controller] 距离锥桶 < 13m，开始准备换道...")

                # 检查左右是否有可行驶车道
                # 注意： get_right_lane() 和 get_left_lane() 是基于行驶方向的
                right_lane_wp = ego_wp.get_right_lane()
                left_lane_wp = ego_wp.get_left_lane()

                # 优先向右换道
                if right_lane_wp and right_lane_wp.lane_type == carla.LaneType.Driving:
                    print("[Controller] 检测到右侧车道可用，命令TM向右换道。")
                    self.tm.force_lane_change(self.ego, True)  # True for right
                    self.state = "MONITORING_CHANGE"
                    self.lane_change_start_wp = ego_wp
                    self.lane_change_timeout = 100  # 设置5秒超时 (100 * 0.05s)
                elif left_lane_wp and left_lane_wp.lane_type == carla.LaneType.Driving:
                    print("[Controller] 检测到左侧车道可用，命令TM向左换道。")
                    self.tm.force_lane_change(self.ego, False)  # False for left
                    self.state = "MONITORING_CHANGE"
                    self.lane_change_start_wp = ego_wp
                    self.lane_change_timeout = 100
                else:
                    print("[Controller] 警告：没有可用的相邻车道！TM将自行决策。")
                    self.state = "MANEUVER_COMPLETE"  # 没地方可换，任务结束

        # 状态2: 已发出换道指令，现在监控是否完成
        elif self.state == "MONITORING_CHANGE":
            self.lane_change_timeout -= 1
            # 检查车道ID是否发生变化
            if self.lane_change_start_wp and ego_wp.lane_id != self.lane_change_start_wp.lane_id:
                print(
                    f"[Controller] ✅ 换道成功! 当前车道ID: {ego_wp.lane_id}, 初始车道ID: {self.lane_change_start_wp.lane_id}")
                self.state = "MANEUVER_COMPLETE"
            # 检查是否超时
            elif self.lane_change_timeout <= 0:
                print("[Controller] ❌ 换道超时，但车道ID未改变。任务结束。")
                self.state = "MANEUVER_COMPLETE"

def spawn_ego_upstream_lane_center(env: HighwayEnv) -> Tuple[carla.Actor, carla.Waypoint]:
    """
    【带诊断信息的版本】
    在第一个锥桶后方生成EGO，并打印详细的执行步骤。
    """
    print("\n--- [EGO 生成诊断 START] ---")
    world = env.world
    amap = world.get_map()
    ego_bp = get_ego_blueprint(world)

    first_tf = env.get_first_cone_transform()

    if first_tf is not None:
        print(f"1. 成功获取到第一个锥桶的位置: {first_tf.location}")
        wp = amap.get_waypoint(first_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if wp is not None:
            print(f"2. 成功在锥桶位置附近找到可行驶车道的路点: {wp.transform.location}")
            for back in [37.0, 38.0, 39.0]:
                print(f"3. 尝试在路点后方 {back}米 处寻找生成点...")
                prevs = wp.previous(back)

                if prevs:
                    spawn_wp = prevs[0]
                    print(f"   - 找到候选路点: {spawn_wp.transform.location}")
                    tf_location = spawn_wp.transform.location + carla.Location(z=0.5)
                    tf_rotation = spawn_wp.transform.rotation
                    tf = carla.Transform(tf_location, tf_rotation)

                    ego = world.try_spawn_actor(ego_bp, tf)
                    if ego:
                        env.set_ego(ego)
                        world.tick()  # 等待一帧让车辆稳定下来
                        ego_current_wp = amap.get_waypoint(ego.get_location(), project_to_road=True,
                                                           lane_type=carla.LaneType.Driving)
                        print(f"   ✅ [成功] 车辆已在后方 {back}米 处创建！")
                        print("--- [EGO 生成诊断 END] ---\n")
                        return ego, ego_current_wp
                    else:
                        print(f"   ❌ [失败] 生成失败。该位置可能被占用或无效。")
                else:
                    print(f"   - [跳过] 未能找到后方 {back}米 处的路点（可能道路太短）。")
        else:
            print("2. ❌ [失败] 在锥桶位置附近未能找到可行驶车道的路点。")
    else:
        print("1. ❌ [失败] 未能获取到第一个锥桶的位置。可能是场景生成失败。")

    print("\n[后备方案] 首选方案失败，现在尝试使用地图默认生成点...")
    spawns = amap.get_spawn_points()
    random.shuffle(spawns)
    for i, tf in enumerate(spawns[:10]):
        print(f"[后备方案] 尝试默认点 #{i + 1}...")
        tf.location.z += 0.20
        ego = world.try_spawn_actor(ego_bp, tf)
        if ego:
            env.set_ego(ego)
            world.tick()
            ego_current_wp = amap.get_waypoint(ego.get_location(), project_to_road=True,
                                               lane_type=carla.LaneType.Driving)
            print(f"   ✅ [成功] 车辆已在默认点创建！")
            print("--- [EGO 生成诊断 END] ---\n")
            return ego, ego_current_wp

    print("--- [EGO 生成诊断 END] ---\n")
    raise RuntimeError("所有方案都已尝试，未能生成EGO。请检查上面的诊断日志确定失败环节。")


def set_spectator_above_start_point(
        world: carla.World,
        start_transform: carla.Transform,
        height: float = 35.0,
        distance_behind: float = 30.0,
        pitch: float = -45.0
):
    """
    将观察者视角（Spectator）设置在指定变换位置的后上方，并固定。

    Args:
        world: CARLA 世界对象。
        start_transform: 自车（EGO）的起始变换，包含位置和旋转。
        height (float): 相机在起始点正上方的高度（米）。
        distance_behind (float): 相机在起始点后方的水平距离（米）。
        pitch (float): 相机的俯仰角（度），负数表示向下看。
    """
    # 1. 获取观察者对象
    spectator = world.get_spectator()

    # 2. 计算相机的位置
    #    a) 获取车辆的“前方”单位向量
    forward_vector = start_transform.get_forward_vector()

    #    b) 将车辆起始位置向后移动 distance_behind 米，向上移动 height 米
    #       注意：减去一个“前方”向量等于向“后方”移动
    camera_location = (
            start_transform.location
            - forward_vector * distance_behind
            + carla.Location(z=height)
    )

    # 3. 计算相机的旋转
    #    相机的水平朝向(yaw)与车辆一致，俯仰角(pitch)设为指定的俯视角度
    camera_rotation = carla.Rotation(
        pitch=pitch,
        yaw=start_transform.rotation.yaw,
        roll=0.0
    )

    # 4. 构建最终的变换并应用到观察者
    final_transform = carla.Transform(camera_location, camera_rotation)
    spectator.set_transform(final_transform)
    print(f"[Spectator] 观察者视角已固定在 {camera_location}")


# ====== 主程序 ======
def main():
    env = HighwayEnv(host="127.0.0.1", port=2000, sync=True, fixed_dt=0.05).connect()
    logger = None
    try:
        env.setup_scene(
            num_cones=5, step_forward=3.0, step_right=0.35,
            z_offset=0.0, min_gap_from_junction=15.0,
            grid=5.0, set_spectator=True
        )

        # 1. 先生成自车，并获取其准确的初始路点
        ego, ego_wp = spawn_ego_upstream_lane_center(env)
        if ego_wp is None:
            raise RuntimeError("无法为已生成的Ego车辆找到有效的路点。")
        set_spectator_above_start_point(env.world, ego_wp.transform)

        # 2. 获取 Traffic Manager 并为EGO启用自动驾驶
        collector = ImitationDataCollector(env.client, ego)
        tm = env.client.get_trafficmanager()  # 通常TM在端口8000
        tm_port = tm.get_port()
        print(f"[Main] 为EGO车辆启用自动驾驶，注册到TM端口 {tm_port}...")
        ego.set_autopilot(True, tm_port)  # 正确使用带端口的 set_autopilot

        # 3. 生成周围交通流
        idp = 0.0  # 这里切换周围交通参与者的密度  行为还要改
        scenemanager = SceneManager(ego_wp, idp)
        scenemanager.gen_traffic_flow(env.world, ego_wp)

        # 4. 初始化控制器和记录器
        first_cone_tf = env.get_first_cone_transform()
        last_cone_tf = env.get_last_cone_transform()
        if first_cone_tf is None:
            raise RuntimeError("场景中没有找到第一个锥桶，无法初始化控制器。")

        controller = ExpertController(ego, last_cone_tf, tm)
        start_location = ego.get_location()
        stall_timer = 0.0
        max_frames = 2000

        # 4. 主循环：运行仿真、控制车辆并记录数据
        for frame in range(max_frames):
            env.world.tick()  # 等待服务器更新

            # 从控制器获取控制指令
            controller.run_step()

            actual_control = ego.get_control()
            if collector.has_collided:
                print("[Main] 检测到碰撞发生！立即停止本次数据采集。")
                break  # 中断循环

                # 如果没有碰撞，才调用tick函数记录数据
            collector.tick(actual_control)

            # 如果进入autopilot模式，可以提前结束或者再运行一段时间后结束
            distance_traveled = ego.get_location().distance(start_location)
            if distance_traveled > 40.0:
                print(f"[Main] 停止条件触发: 行驶距离 {distance_traveled:.2f}m > 40m。")
                break

            current_speed = ego.get_velocity().length()
            if current_speed < 0.1:
                stall_timer += env.fixed_dt
            else:
                stall_timer = 0.0

            if stall_timer >= 5.0:
                print(f"[Main] 停止条件触发: 停车时间 {stall_timer:.2f}s >= 5s。")
                break

        # 检查是否因为达到最大帧数而退出
        if frame == max_frames - 1:
            print(f"[Main] 安全警告: 已达到最大帧数 {max_frames}，强制结束仿真。")

        print("\n[Main] 仿真循环结束。")

    except KeyboardInterrupt:
        print("\n[Stop] 手动退出。")
    except Exception as e:
        print(f"\n[Error] 发生错误: {e}")
    finally:

        print("\n[Debug] 程序执行结束，正在进入 finally 清理流程...")  # <--- 新增路标1

        if collector is not None:
            print("[Debug] Collector存在，准备保存数据和销毁传感器。")  # <--- 新增路標2

            # 对保存和销毁操作也进行try-except封装，防止其中一个失败影响另一个
            try:
                collector.save_to_disk()
            except Exception as e:
                print(f"[Debug] 错误：在调用 save_to_disk() 时发生异常: {e}")

            try:
                collector.destroy_sensors()
            except Exception as e:
                print(f"[Debug] 错误：在调用 destroy_sensors() 时发生异常: {e}")

            print("[Debug] 数据保存和销毁指令已调用。")  # <--- 新增路標3
        else:
            print("[Debug] Collector 未被初始化，跳过保存和销毁步骤。")

        try:
            if ego and ego.is_alive:
                ego.set_autopilot(False)
            if env:
                env.close()
            print("[Debug] 环境已成功关闭。")  # <--- 新增路標4
        except Exception as e:
            print(f"关闭环境时发生错误: {e}")


if __name__ == "__main__":
    main()