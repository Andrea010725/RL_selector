import carla
import numpy as np
import time


class RealTimeCarlaProvider:
    """
    一个实时数据提供者，用于从正在运行的CARLA仿真中提取和格式化数据。
    """

    def __init__(self, host='localhost', port=2000):
        """初始化，连接到CARLA服务器"""
        # --- 配置参数 ---

        self.DETECTION_RADIUS = 50.0  # 只考虑自车周围50米范围内的物体
        self.LANE_DETECTION_RADIUS = 50.0  # 提取自车周围50米范围内的车道线
        self.FUTURE_FRAMES_PRED = 30  # 需要预测的未来轨迹帧数
        self.TIMESTEP = 0.1  # 假设的仿真步长，用于预测 (10 Hz)

    @staticmethod
    def get_actor_class(actor):
        """获取actor的类别名称"""
        if 'vehicle' in actor.type_id:
            return 'car'
        if 'walker' in actor.type_id:
            return 'pedestrian'
        return 'other'

    @staticmethod
    def world_to_ego_coords(points, ego_transform_matrix):
        """将世界坐标点或向量转换到自车坐标系下"""
        ego_matrix_inv = np.linalg.inv(ego_transform_matrix)
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        ego_points_h = points_h @ ego_matrix_inv.T
        return ego_points_h[:, :3]

    def get_map_lanes(self, ego_transform):
        """
        实时提取自车周围的车道线和道路边界。
        返回一个列表，其中每个元素都是一个包含点坐标和类型的字典。
        """
        ego_location = ego_transform.location
        vectorized_lanes = []

        # 简化提取逻辑：在自车周围一定半径内采样路点
        waypoints = self.map.get_waypoint(ego_location).next_until_lane_end(self.LANE_DETECTION_RADIUS)

        processed_lanes = set()  # 防止重复处理

        for wp in waypoints:
            # 提取左侧车道线
            left_lane = wp.get_left_lane()
            if left_lane and left_lane.lane_id not in processed_lanes:
                processed_lanes.add(left_lane.lane_id)
                lane_wps = left_lane.next_until_lane_end(self.LANE_DETECTION_RADIUS)
                points = np.array(
                    [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in lane_wps])
                if points.shape[0] > 1:
                    vectorized_lanes.append({'type': 'lane_divider', 'points_world': points})

            # 提取右侧车道线（即当前车道本身）
            if wp.lane_id not in processed_lanes:
                processed_lanes.add(wp.lane_id)
                lane_wps = wp.next_until_lane_end(self.LANE_DETECTION_RADIUS)
                points = np.array(
                    [[w.transform.location.x, w.transform.location.y, w.transform.location.z] for w in lane_wps])
                if points.shape[0] > 1:
                    vectorized_lanes.append({'type': 'lane_divider', 'points_world': points})

        # 将所有点转换到自车坐标系
        ego_matrix = np.array(ego_transform.get_matrix())
        for lane in vectorized_lanes:
            lane['points_ego'] = self.world_to_ego_coords(lane['points_world'], ego_matrix)

        return vectorized_lanes

    def get_current_frame_data(self):
        """
        获取当前帧的所有关键数据，并格式化为输入形式。
        这是一个核心函数，用于实时调用。
        """
        # 1. 找到自车并获取其变换矩阵
        ego_vehicle = None
        for actor in self.world.get_actors():
            if actor.attributes.get('role_name') == 'ego':
                ego_vehicle = actor
                break

        if ego_vehicle is None:
            raise RuntimeError("Ego vehicle not found in the simulation. Make sure to spawn one with role_name='ego'.")

        ego_transform = ego_vehicle.get_transform()
        ego_matrix = np.array(ego_transform.get_matrix())

        # 初始化样本字典
        sample = {
            'timestamp': self.world.get_snapshot().timestamp.elapsed_seconds,
            'ego_transform': ego_transform,
            'objects': {},
            'map_lanes': self.get_map_lanes(ego_transform)
        }

        # 2. 遍历所有actor，提取周边物体信息
        actor_list = self.world.get_actors().filter('vehicle.*')  # 或者 'vehicle.*|walker.*'
        for actor in actor_list:
            if actor.id == ego_vehicle.id:
                continue

            # 过滤掉远处的物体
            actor_location = actor.get_location()
            if actor_location.distance(ego_transform.location) > self.DETECTION_RADIUS:
                continue

            # 提取位置、尺寸、类别
            pos_world = np.array([[actor_location.x, actor_location.y, actor_location.z]])
            pos_ego = self.world_to_ego_coords(pos_world, ego_matrix)[0]

            # 提取速度
            vel_world = actor.get_velocity()
            vel_vec_world = np.array([[vel_world.x, vel_world.y, vel_world.z]])
            # 速度变换只受旋转影响
            ego_rotation_only_matrix = ego_transform
            ego_rotation_only_matrix.location = carla.Location(0, 0, 0)  # 清除平移
            vel_ego = self.world_to_ego_coords(vel_vec_world, np.array(ego_rotation_only_matrix.get_matrix()))[0]

            # 预测未来轨迹 (注意：这是预测值，不是真值)
            future_trajectory = []
            current_pos = pos_ego[:2]  # 使用自车系下的二维坐标
            current_vel = vel_ego[:2]
            for _ in range(self.FUTURE_FRAMES_PRED):
                current_pos += current_vel * self.TIMESTEP
                future_trajectory.append(current_pos.copy())

            sample['objects'][actor.id] = {
                'class': self.get_actor_class(actor),
                'position_ego': pos_ego,  # [x, y, z]
                'size': [  # [length, width, height]
                    actor.bounding_box.extent.x * 2,
                    actor.bounding_box.extent.y * 2,
                    actor.bounding_box.extent.z * 2,
                ],
                'velocity_ego': vel_ego,  # [vx, vy, vz]
                'predicted_future_trajectory_ego': np.array(future_trajectory)  # [FUTURE_FRAMES, 2]
            }

        return sample

    def collect_data(self):
        """主数据采集循环"""
        total_frames = int(DATA_COLLECTION_SECONDS / SIM_TIMESTEP)

        # 1. 运行仿真，记录所有actor在每一帧的状态
        print("Stage 1: Running simulation and recording all actor states...")
        live_actors = {actor.id: actor for actor in self.actor_list}
        all_frames_data = []

        for frame in range(total_frames):
            self.world.tick()
            current_frame_data = {}
            ego_transform = self.ego_vehicle.get_transform()

            for actor_id, actor in live_actors.items():
                if not actor.is_alive: continue
                current_frame_data[actor_id] = {
                    'transform': actor.get_transform(),
                    'velocity': actor.get_velocity(),
                    'bounding_box': actor.bounding_box,
                    'class': self.get_actor_class(actor)
                }
            all_frames_data.append(current_frame_data)

            if frame % 100 == 0:
                print(f"  ... collected frame {frame}/{total_frames}")

        # 2. 离线处理记录的数据，生成样本
        print("\nStage 2: Processing recorded data to generate samples...")
        final_dataset = []

        # 窗口滑动范围，确保有足够的历史和未来数据
        for frame_idx in range(HISTORY_FRAMES, total_frames - FUTURE_FRAMES):
            sample = {'frame_id': frame_idx, 'objects': {}, 'map_lanes': []}

            # 获取当前帧的自车信息作为参考系
            ego_id = self.ego_vehicle.id
            if ego_id not in all_frames_data[frame_idx]: continue  # 如果自车消失则跳过

            ego_data = all_frames_data[frame_idx][ego_id]
            ego_transform = ego_data['transform']
            ego_location = ego_transform.location

            # --- a. 提取周边物体信息 ---
            for actor_id, actor_data in all_frames_data[frame_idx].items():
                if actor_id == ego_id: continue  # 跳过自车

                actor_location = actor_data['transform'].location
                if ego_location.distance(actor_location) > DETECTION_RADIUS:
                    continue

                # 位置、尺寸、类别 (在自车坐标系下)
                pos_world = np.array([[actor_location.x, actor_location.y, actor_location.z]])
                pos_ego = self.world_to_ego_coords(pos_world, ego_transform)[0]

                # 速度 (在自车坐标系下)
                vel_world = actor_data['velocity']
                vel_vec_world = np.array([[vel_world.x, vel_world.y, vel_world.z]])
                # 速度变换只受旋转影响
                vel_ego = self.world_to_ego_coords(vel_vec_world, carla.Transform(rotation=ego_transform.rotation))[0]

                # 未来轨迹 (在自车坐标系下)
                future_trajectory = []
                for i in range(1, FUTURE_FRAMES + 1):
                    future_frame_data = all_frames_data[frame_idx + i]
                    if actor_id in future_frame_data:
                        future_pos_world = future_frame_data[actor_id]['transform'].location
                        future_pos_world_np = np.array([[future_pos_world.x, future_pos_world.y, future_pos_world.z]])
                        future_pos_ego = self.world_to_ego_coords(future_pos_world_np, ego_transform)[0]
                        future_trajectory.append(future_pos_ego[:2])  # 只保留 x, y

                # 只有当有完整未来轨迹时才保存该物体
                if len(future_trajectory) == FUTURE_FRAMES:
                    sample['objects'][actor_id] = {
                        'class': actor_data['class'],
                        'position_ego': pos_ego,  # [x, y, z]
                        'size': [  # [length, width, height]
                            actor_data['bounding_box'].extent.x * 2,
                            actor_data['bounding_box'].extent.y * 2,
                            actor_data['bounding_box'].extent.z * 2,
                        ],
                        'velocity_ego': vel_ego,  # [vx, vy, vz]
                        'future_trajectory_ego': np.array(future_trajectory)  # [FUTURE_FRAMES, 2]
                    }

            # --- b. 提取地图信息 ---
            sample['map_lanes'] = self.get_map_lanes(ego_transform)

            final_dataset.append(sample)
            if frame_idx % 100 == 0:
                print(f"  ... packaged sample for frame {frame_idx}")

        return final_dataset

    def cleanup(self):
        """销毁所有actor并恢复世界设置"""
        print("\nCleaning up actors...")
        self.world.apply_settings(self.original_settings)  # 恢复异步模式
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        print(f"Destroyed {len(self.actor_list)} actors.")