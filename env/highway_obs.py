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
    "vehicle.tesla.cybertruck",
    "vehicle.audi.tt",
    "vehicle.audi.a2",
    "vehicle.bmw.grandtourer",
    "vehicle.lincoln.mkz2017",
    "vehicle.mercedes.coupe",
    "vehicle.nissan.micra",
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

def place_right_shifted_cones(
    world: carla.World,
    start_wp: carla.Waypoint,
    num_cones: int,
    step_forward: float,
    step_right: float,
    z_offset: float = 0.0,
    ensure_cross_right_edge: bool = True,
    margin: float = 0.05,
) -> Tuple[List[carla.Actor], Optional[carla.Transform]]:
    if num_cones < 2:
        raise ValueError("num_cones 至少为 2。")
    bp = get_cone_blueprint(world)
    lane_half_width = start_wp.lane_width * 0.5
    dx_base = -0.5 * start_wp.lane_width
    if ensure_cross_right_edge:
        need = (lane_half_width + margin) - dx_base
        min_step_right = max(need / (num_cones - 1), 1e-3)
        if step_right < min_step_right:
            print(f"[提示] step_right={step_right:.3f} 过小，自动放大为 {min_step_right:.3f}")
            step_right = min_step_right

    actors: List[carla.Actor] = []
    first_cone_tf: Optional[carla.Transform] = None

    current_wp = start_wp.next(max(0.25, min(0.5, step_forward * 0.2)))[0]
    for i in range(num_cones):
        if i > 0:
            next_wps = current_wp.next(step_forward)
            if not next_wps:
                print(f"[警告] 前方 {step_forward} m 无有效 waypoint，停止于 i={i}.")
                break
            current_wp = next_wps[0]
        wp = current_wp
        yaw = wp.transform.rotation.yaw
        center_loc = wp.transform.location
        dx_right = dx_base + i * step_right
        cone_loc = shift_location(center_loc, yaw_deg=yaw, dx_right=dx_right, dy_forward=0.0, dz=z_offset)
        cone_tf = carla.Transform(location=cone_loc, rotation=carla.Rotation(
            pitch=wp.transform.rotation.pitch, yaw=yaw, roll=wp.transform.rotation.roll))
        try:
            cone = world.spawn_actor(bp, cone_tf)
            with contextlib.suppress(Exception):
                cone.set_simulate_physics(False)
            actors.append(cone)
            if first_cone_tf is None:
                first_cone_tf = cone_tf
        except RuntimeError as e:
            print(f"[错误] i={i} 处生成锥桶失败：{e}")

    return actors, first_cone_tf

def set_spectator_to_view_first_cone(
    world: carla.World,
    first_cone_tf: carla.Transform,
    view_distance: float = 18.0,
    height: float = 15.0,
    pitch_deg: float = -20.0,
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
    
def set_spectator_follow_actor(
    world: carla.World,
    actor: carla.Actor,
    mode: str = "chase",       # "chase" 追尾；"top" 俯视
    distance: float = 18.0,    # 车后距离
    height: float = 7.0,       # 相机高度
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

    def setup_scene(
        self,
        num_cones: int = 10,
        step_forward: float = 3.0,
        step_right: float = 0.25,
        z_offset: float = 0.0,
        min_gap_from_junction: float = 15.0,
        grid: float = 5.0,
        set_spectator: bool = True,
    ):
        assert self.world is not None
        start_wp = pick_random_start_waypoint(self.world, min_gap_from_junction=min_gap_from_junction, grid=grid, max_tries=300)
        cones, first_tf = place_right_shifted_cones(
            world=self.world,
            start_wp=start_wp,
            num_cones=num_cones,
            step_forward=step_forward,
            step_right=step_right,
            z_offset=z_offset,
            ensure_cross_right_edge=True,
            margin=0.05,
        )
        self._actors.extend(cones)
        if self.sync:
            for _ in range(3): self.world.tick()
        else:
            self.world.wait_for_tick()
        self._first_cone_tf = first_tf
        if set_spectator and first_tf is not None:
            set_spectator_to_view_first_cone(self.world, first_tf)

    # === 新增：提供第一个锥桶位姿给 agent 用来计算 EGO 的生成位置 ===
    def get_first_cone_transform(self) -> Optional[carla.Transform]:
        return self._first_cone_tf

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
            for a in self._actors:
                with contextlib.suppress(Exception):
                    a.destroy()
        finally:
            self._actors.clear()
            self.ego = None
            if self._sync_cm is not None:
                self._sync_cm.__exit__(None, None, None)
                self._sync_cm = None
