# env/highway_obs.py
import math
import contextlib
from typing import List, Optional, Tuple, Dict, Any
import random
import carla

CONE_BP_CANDIDATES = [
    "static.prop.trafficcone01",
    "static.prop.trafficcone",
    "static.prop.trafficcone02",
    "static.prop.cone",
]

EGO_BP_CANDIDATES = [
    "vehicle.tesla.model3",
]

@contextlib.contextmanager
def carla_sync_mode(client: carla.Client, world: carla.World, enabled: bool):
    if not enabled:
        yield
        return
    original_settings = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        yield
    finally:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)

def get_cone_blueprint(world: carla.World) -> carla.ActorBlueprint:
    lib = world.get_blueprint_library()
    last_err: Optional[Exception] = None
    for name in CONE_BP_CANDIDATES:
        try:
            return lib.find(name)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"找不到可用的锥桶蓝图，尝试过：{CONE_BP_CANDIDATES}. 最后错误：{last_err}")

def get_ego_blueprint(world: carla.World) -> carla.ActorBlueprint:
    lib = world.get_blueprint_library()
    for name in EGO_BP_CANDIDATES:
        try:
            bp = lib.find(name)
            if bp.has_attribute("color"):
                reds = [c for c in bp.get_attribute("color").recommended_values if "255,0,0" in c or "red" in c.lower()]
                bp.set_attribute("color", reds[0] if reds else "255,0,0")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
            return bp
        except Exception:
            continue
    for bp in lib.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels") and bp.get_attribute("number_of_wheels").as_int() == 4:
            if bp.has_attribute("color"):
                bp.set_attribute("color", "255,0,0")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
            return bp
    raise RuntimeError("未找到可用车辆蓝图。")

def right_unit_vector_from_yaw(yaw_deg: float) -> carla.Vector3D:
    rad = math.radians(yaw_deg)
    return carla.Vector3D(x=math.sin(rad), y=-math.cos(rad), z=0.0)

def forward_unit_vector_from_yaw(yaw_deg: float) -> carla.Vector3D:
    rad = math.radians(yaw_deg)
    return carla.Vector3D(x=math.cos(rad), y=math.sin(rad), z=0.0)

def shift_location(loc: carla.Location, yaw_deg: float, dx_right: float, dy_forward: float, dz: float = 0.0) -> carla.Location:
    r = right_unit_vector_from_yaw(yaw_deg)
    f = forward_unit_vector_from_yaw(yaw_deg)
    return carla.Location(
        x=loc.x + r.x * dx_right + f.x * dy_forward,
        y=loc.y + r.y * dx_right + f.y * dy_forward,
        z=loc.z + dz,
    )

def is_near_junction(wp: carla.Waypoint, dist: float = 15.0, step: float = 1.0) -> bool:
    cur = wp; traveled = 0.0
    while traveled < dist:
        nxt = cur.next(step)
        if not nxt: break
        cur = nxt[0]; traveled += step
        if cur.is_junction: return True
    cur = wp; traveled = 0.0
    while traveled < dist:
        prv = cur.previous(step)
        if not prv: break
        cur = prv[0]; traveled += step
        if cur.is_junction: return True
    return False

def pick_random_start_waypoint(world: carla.World, min_gap_from_junction: float = 15.0, grid: float = 5.0, max_tries: int = 300) -> carla.Waypoint:
    amap = world.get_map()
    candidates = [wp for wp in amap.generate_waypoints(grid) if wp.lane_type == carla.LaneType.Driving]
    if not candidates:
        raise RuntimeError("地图上没有可用的 Driving waypoints。")
    random.shuffle(candidates)
    tries = 0
    for wp in candidates:
        tries += 1
        if (not wp.is_junction) and (not is_near_junction(wp, dist=min_gap_from_junction, step=1.0)):
            return wp
        if tries >= max_tries:
            break
    for wp in candidates:
        if not wp.is_junction:
            return wp
    return candidates[0]


# 假设 get_cone_blueprint 和 shift_location 是您项目中已定义的辅助函数
# 为了让代码可独立运行，我这里补充一个虚拟的实现
def get_cone_blueprint(world: carla.World) -> carla.ActorBlueprint:
    return world.get_blueprint_library().find('static.prop.trafficcone01')


def shift_location(location: carla.Location, yaw_deg: float, dx_right: float, dy_forward: float,
                   dz: float) -> carla.Location:
    yaw_rad = math.radians(yaw_deg)
    final_loc = carla.Location(
        x=location.x + dy_forward * math.cos(yaw_rad) - dx_right * math.sin(yaw_rad),
        y=location.y + dy_forward * math.sin(yaw_rad) + dx_right * math.cos(yaw_rad),
        z=location.z + dz
    )
    return final_loc

def set_spectator_to_view_first_cone(
    world: carla.World,
    first_cone_tf: carla.Transform,
    view_distance: float = 25.0,
    height: float = 18.0,
    pitch_deg: float = -15.0,
):
    spec = world.get_spectator()
    yaw = first_cone_tf.rotation.yaw
    back_vec = forward_unit_vector_from_yaw(yaw)
    cam_loc = carla.Location(
        x=first_cone_tf.location.x - back_vec.x * view_distance,
        y=first_cone_tf.location.y - back_vec.y * view_distance,
        z=first_cone_tf.location.z + height,
    )
    cam_rot = carla.Rotation(pitch=pitch_deg, yaw=yaw, roll=0.0)
    spec.set_transform(carla.Transform(cam_loc, cam_rot))

def set_spectator_fixed(world: carla.World,
    ego: carla.Actor,
    back: float = 28.0,      # 相机沿车头反向后退的距离（米）
    height: float = 7.0,     # 相机抬高（米）
    side_offset: float = 0.0,# 右侧偏移（米），正数=向右，负数=向左
    pitch_deg: float = -12.0,# 俯仰角（负值俯视）
    look_at_roof: bool = True, # 自动微调俯仰角去看向车顶（更稳）
):
    """
    把 spectator 固定在“自车后上方”，朝向自车前进方向。
    - 不做任何后续更新；在初始化时调用一次即可。
    - 右侧偏移 side_offset 可以让你看到一点点车身。
    """
    spec = world.get_spectator()
    tf = ego.get_transform()

    # 车头朝向（度→弧度）
    yaw = tf.rotation.yaw
    rad = math.radians(yaw)

    # 前/右单位向量（CARLA 坐标系：x 前进，y 左→右为正；右向 = (sin,yaw, -cos,yaw)）
    fx, fy = math.cos(rad), math.sin(rad)
    rx, ry = math.sin(rad), -math.cos(rad)

    # 计算相机位置：在自车后方 back 米、右侧 side_offset 米、上方 height 米
    cam_loc = carla.Location(
        x=tf.location.x - fx * back + rx * side_offset,
        y=tf.location.y - fy * back + ry * side_offset,
        z=tf.location.z + height,
    )

    # 计算朝向：默认沿自车前进方向；可选自动瞄准“车顶”
    if look_at_roof:
        # 目标看向自车车顶附近（略高 1.5m）
        tgt = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 1.5)
        vx, vy, vz = (tgt.x - cam_loc.x), (tgt.y - cam_loc.y), (tgt.z - cam_loc.z)
        yaw_cam = math.degrees(math.atan2(vy, vx))          # 朝目标的平面方位角
        dist_xy = max(1e-6, math.hypot(vx, vy))
        pitch_cam = -math.degrees(math.atan2(vz, dist_xy))  # 向下为负
    else:
        yaw_cam = yaw
        pitch_cam = pitch_deg

    cam_rot = carla.Rotation(pitch=pitch_cam, yaw=yaw_cam, roll=0.0)
    spec.set_transform(carla.Transform(cam_loc, cam_rot))

    
def set_spectator_follow_actor(
    world: carla.World,
    actor: carla.Actor,
    mode: str = "chase",       # "chase" 追尾；"top" 俯视
    distance: float = 23.0,    # 车后距离
    height: float = 9.0,       # 相机高度
    pitch_deg: float = -12.0,  # 俯仰角
    yaw_offset: float = 0.0,   # 相对车头的偏航偏置
    side_offset: float = 2.0   # 向右侧偏一点，便于看清车身
):
    spec = world.get_spectator()
    tf = actor.get_transform()
    yaw = tf.rotation.yaw
    rad = math.radians(yaw)
    fx, fy = math.cos(rad), math.sin(rad)       # forward
    rx, ry = math.sin(rad), -math.cos(rad)      # right

    if mode == "top":
        cam_loc = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + height)
        cam_rot = carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0)
    else:
        cam_loc = carla.Location(
            x=tf.location.x - fx*distance + rx*side_offset,
            y=tf.location.y - fy*distance + ry*side_offset,
            z=tf.location.z + height
        )
        cam_rot = carla.Rotation(pitch=pitch_deg, yaw=yaw + yaw_offset, roll=0.0)

    spec.set_transform(carla.Transform(cam_loc, cam_rot))


class HighwayEnv:
    def __init__(self, host: str="127.0.0.1", port: int=2000, sync: bool=True, fixed_dt: float=0.05):
        self.host = host
        self.port = port
        self.sync = sync
        self.fixed_dt = fixed_dt
        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self._actors: List[carla.Actor] = []
        self.ego: Optional[carla.Actor] = None
        self._sync_cm = None
        self._first_cone_tf: Optional[carla.Transform] = None

    def connect(self, timeout: float=5.0):
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self._sync_cm = carla_sync_mode(self.client, self.world, enabled=self.sync)
        self._sync_cm.__enter__()
        return self

    def place_cones_conditionally_behind(
            self,
            world: carla.World,
            start_wp: carla.Waypoint,
            num_cones: int = 10,
            step_behind: float = 3.0,
            step_lateral_per_cone: float = 0.35,
            z_offset: float = 0.0,
            lane_margin: float = 0.25,
    ) -> Tuple[List[carla.Actor], Optional[carla.Transform], Optional[carla.Transform]]:
        """
        基于 lane_id 正负（同向/反向）与“是否可行驶(Driving)”的判定来放置锥桶。
        规则：
          1) 如果某一侧不是 Driving，则封该侧，引导车辆向另一侧。
          2) 若两侧都是 Driving，则优先封“反向侧”，引导车辆留在同向侧。
          3) 若两侧同为同向或同为反向（少见），固定封左侧（可按需改成随机）。
        摆放方式：
          - 从被封侧的车道边缘起步（贴边但预留 lane_margin），
          - 每个锥桶横向推进 step_lateral_per_cone，纵向沿 previous(step_behind) 往后摆，
          - 对贴边点 spawn 失败，做“向内（朝0）微调重试”，保证第一个锥桶尽量贴边。
        返回: (生成的actor列表, 第一个锥桶Transform, 最后一个锥桶Transform)
        """
        cone_blueprint = world.get_blueprint_library().find("static.prop.trafficcone01")
        if not cone_blueprint:
            print("错误：找不到锥桶蓝图 'static.prop.trafficcone01'")
            return [], None, None

        def is_driving(wp: Optional[carla.Waypoint]) -> bool:
            return bool(wp and wp.lane_type == carla.LaneType.Driving)

        def same_direction(a: carla.Waypoint, b: Optional[carla.Waypoint]) -> bool:
            """仅用 lane_id 符号判断同/反向；b 为空则返回 False。"""
            try:
                return bool(b and int(a.lane_id) * int(b.lane_id) > 0)
            except Exception:
                return False

        cones_spawned: List[carla.Actor] = []
        first_cone_transform: Optional[carla.Transform] = None
        last_cone_transform: Optional[carla.Transform] = None

        current_wp = start_wp

        for i in range(num_cones):
            if not current_wp:
                break

            wp_tf = current_wp.transform
            right_vector = wp_tf.get_right_vector()  # 只用于 offset->世界坐标（正=向右）
            half_lane_width = current_wp.lane_width * 0.5

            # === 基于 lane_id / Driving 判定“封哪一侧” ===
            left_lane_wp = current_wp.get_left_lane()
            right_lane_wp = current_wp.get_right_lane()

            left_is_drv = is_driving(left_lane_wp)
            right_is_drv = is_driving(right_lane_wp)
            left_same = same_direction(current_wp, left_lane_wp) if left_is_drv else False
            right_same = same_direction(current_wp, right_lane_wp) if right_is_drv else False

            # block_side: "left" 表示从左边缘起步并向右推进（封左，引右）
            #             "right" 表示从右边缘起步并向左推进（封右，引左）
            if left_is_drv and not right_is_drv:
                block_side = "right"  # 右侧不可行驶 → 封右
            elif right_is_drv and not left_is_drv:
                block_side = "left"  # 左侧不可行驶 → 封左
            elif left_is_drv and right_is_drv:
                # 两侧都可行驶：优先封反向侧
                if left_same and not right_same:
                    block_side = "right"  # 右为反向 → 封右
                elif right_same and not left_same:
                    block_side = "left"  # 左为反向 → 封左
                else:
                    block_side = "left"  # 都同向或都反向：固定封左（可改为随机）
            else:
                block_side = "left"  # 两侧都不可行驶（边缘/窄道），兜底封左

            # === 按封堵侧计算“起点贴边 + 推进方向” ===
            if block_side == "left":
                start_offset_signed = -(half_lane_width - lane_margin)  # 左边缘（负）
                step_dir = +1.0  # 向右推进   +
            else:  # "right"
                start_offset_signed = +(half_lane_width - lane_margin)  # 右边缘（正）
                step_dir = -1.0  # 向左推进    -

            progression_offset = (i * step_lateral_per_cone) * step_dir
            desired_offset_signed = start_offset_signed + progression_offset

            # 夹逼到边界内：正=右边缘，负=左边缘
            max_pos = +(half_lane_width - lane_margin)
            max_neg = -(half_lane_width - lane_margin)
            actual_offset_signed = max(max_neg, min(max_pos, desired_offset_signed))

            # === 向内微调重试（朝 0 靠拢：左(-)→+；右(+)→-）===
            cone_actor = None
            cone_transform = None

            nudge_dir = +1.0 if actual_offset_signed < 0.0 else -1.0
            NUDGE_STEP = 0.18
            NUDGE_TRY = 6

            for k in range(0, NUDGE_TRY + 1):
                offset_try = actual_offset_signed + k * nudge_dir * NUDGE_STEP
                offset_try = max(max_neg, min(max_pos, offset_try))

                rv = wp_tf.get_right_vector()
                left_vector = carla.Vector3D(-rv.x, -rv.y, -rv.z)  # 显式取反

                cone_loc_try = wp_tf.location + left_vector * offset_try
                cone_loc_try.z += (z_offset if z_offset != 0.0 else 0.10)  # 稍抬高，减少地面穿插概率
                cone_tf_try = carla.Transform(cone_loc_try, wp_tf.rotation)

                actor_try = world.try_spawn_actor(cone_blueprint, cone_tf_try)
                if actor_try:
                    cone_actor = actor_try
                    cone_transform = cone_tf_try
                    break

            # 成功则记录
            if cone_actor:
                cones_spawned.append(cone_actor)
                if first_cone_transform is None:
                    first_cone_transform = cone_transform
                last_cone_transform = cone_transform

                # （可选）调试打印
                # print(f"[Cone #{i}] block={block_side} left_drv={left_is_drv}({left_same}) "
                #       f"right_drv={right_is_drv}({right_same}) start={start_offset_signed:+.2f} "
                #       f"used={offset_try:+.2f} (neg=左,pos=右)")
            # 纵向后退到下一排
            prevs = current_wp.previous(step_behind)
            if prevs:
                current_wp = prevs[0]
            else:
                print(f"[警告] 后方 {step_behind} m 无有效 waypoint，停止于第 {i + 1} 个锥桶处。")
                current_wp = None

        return cones_spawned, first_cone_transform, last_cone_transform

    def setup_scene(
            self,
            num_cones: int = 10,
            step_forward: float = 3.0,  # 这个参数在新逻辑中代表 "step_behind"
            step_right: float = 0.35,  # 这个参数在新逻辑中代表 "step_lateral"
            z_offset: float = 0.0,
            min_gap_from_junction: float = 15.0,
            grid: float = 5.0,
            set_spectator: bool = True,
    ):
        assert self.world is not None
        # 步骤1：随机选择起始点 (这部分逻辑不变)
        start_wp = pick_random_start_waypoint(self.world, min_gap_from_junction=min_gap_from_junction, grid=grid,
                                              max_tries=300)

        cones, first_tf, last_tf = self.place_cones_conditionally_behind(
            world=self.world,
            start_wp=start_wp,
            num_cones=num_cones,
            step_behind=step_forward,
            step_lateral_per_cone=step_right,  # 这个值现在控制斜线的“斜率”
            z_offset=z_offset,
            lane_margin=0.25  # 可以微调这个值，控制锥桶离车道线多近
        )

        self._actors.extend(cones)
        if self.sync:
            for _ in range(3): self.world.tick()
        else:
            self.world.wait_for_tick()

        # 注意：这里的 first_tf 现在是“最后方”的那个锥桶的transform
        # 这也符合逻辑，因为车辆应该从这个锥桶后方更远的位置出现
        self._first_cone_tf = first_tf
        self._last_cone_tf = last_tf
        # if set_spectator and first_tf is not None:
        #     # 这个函数可能需要微调，以确保视角能看到整个锥桶队列
        #     # set_spectator_to_view_first_cone(self.world, first_tf)
        #     set_spectator_fixed(self.world, self.ego)


    # === 新增：提供第一个锥桶位姿给 agent 用来计算 EGO 的生成位置 ===
    def get_first_cone_transform(self) -> Optional[carla.Transform]:
        return self._first_cone_tf

    def get_last_cone_transform(self) -> Optional[carla.Transform]:
        return self._last_cone_tf

    def get_cone_actors(self) -> List[carla.Actor]:
        return list(self._cones) if hasattr(self, "_cones") else []

    # === 新增：agent 生成 EGO 后，登记给环境，方便 get_observation/apply_control ===
    def set_ego(self, ego_actor: carla.Actor):
        self.ego = ego_actor
        if ego_actor is not None:
            self._actors.append(ego_actor)

    def get_observation(self) -> Dict[str, Any]:
        assert self.world is not None
        obs: Dict[str, Any] = {}
        if self.ego is not None:
            tr = self.ego.get_transform()
            vel = self.ego.get_velocity()
            obs["ego_pose"] = dict(x=tr.location.x, y=tr.location.y, z=tr.location.z, yaw=float(tr.rotation.yaw))
            speed = (vel.x**2 + vel.y**2 + vel.z**2)**0.5
            obs["ego_v"]   = dict(vx=vel.x, vy=vel.y, vz=vel.z, speed=speed)
            amap = self.world.get_map()
            wp = amap.get_waypoint(tr.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp:
                obs["lane_info"] = dict(lane_width=wp.lane_width, yaw=float(wp.transform.rotation.yaw))
        cones_xyz = []
        for a in self._actors:
            try:
                if "trafficcone" in a.type_id or "static.prop.cone" in a.type_id:
                    loc = a.get_transform().location
                    cones_xyz.append((loc.x, loc.y, loc.z))
            except RuntimeError:
                pass
        obs["cones"] = cones_xyz
        return obs

    def apply_control(self, throttle: float, steer: float, brake: float = 0.0, hand_brake: bool=False, reverse: bool=False):
        if self.ego is None: return
        control = carla.VehicleControl(
            throttle=float(max(0.0, min(1.0, throttle))),
            steer=float(max(-1.0, min(1.0, steer))),
            brake=float(max(0.0, min(1.0, brake))),
            hand_brake=hand_brake,
            reverse=reverse,
        )
        self.ego.apply_control(control)

    def step(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        assert self.world is not None
        if self.sync: self.world.tick()
        else: self.world.wait_for_tick()
        obs = self.get_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def close(self):
        try:
            actor_list = self.world.get_actors()

            for a in actor_list:
                with contextlib.suppress(Exception):
                    a.destroy()
        finally:
            self._actors.clear()
            self.ego = None
            if self._sync_cm is not None:
                self._sync_cm.__exit__(None, None, None)
                self._sync_cm = None
