# agents/rule_based/agent.py
from __future__ import annotations
import math
import random
import sys
from types import SimpleNamespace
from typing import List, Tuple

sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np
import ipdb

# 你的工程内模块
# 注意：确保这里的路径是您本地RL_selector项目的根路径
sys.path.append("/home/ajifang/czw/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.tools import SceneManager
from agents.rule_based.vis_debug import draw_corridor, draw_ego_marker, TelemetryLogger, draw_lane_envelope, \
    annotate_lane_width
from agents.rule_based.vis_debug import *


# ====== 2) 极简 LaneRef：沿同一条驾驶车道采样，提供 xy<->(s,ey) ======
class LaneRef:
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len: float = 500.0):
        pts, wps = [], []
        wp = seed_wp
        guard_ids = (wp.road_id, wp.section_id, wp.lane_id)
        dist = 0.0
        pts.append((wp.transform.location.x, wp.transform.location.y))
        wps.append(wp)
        while dist < max_len:
            nxts = wp.next(step)
            if not nxts:
                break
            wp = nxts[0]
            if (wp.road_id, wp.section_id, wp.lane_id) != guard_ids:
                break
            pts.append((wp.transform.location.x, wp.transform.location.y))
            wps.append(wp)
            dist += step
        self.P = np.asarray(pts, dtype=float)  # [N,2]
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])  # [N]
        tang = np.diff(self.P, axis=0)
        tang = np.vstack([tang, tang[-1]])
        self.tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        self.wps = wps  # 保存Waypoints，便于车道宽获取
        self.step = float(step)

    def _segment_index_and_t(self, x, y):
        P = self.P;
        xy = np.array([x, y], dtype=float)
        v = xy - P[:-1]
        seg = P[1:] - P[:-1]
        seg_len2 = (seg[:, 0] ** 2 + seg[:, 1] ** 2) + 1e-9
        t = np.clip((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:, None]
        dist2 = np.sum((proj - xy[None, :]) ** 2, axis=1)
        i = int(np.argmin(dist2))
        return i, float(t[i]), proj[i]

    def xy2se(self, x: float, y: float):
        i, t, proj = self._segment_index_and_t(x, y)
        s_val = self.s[i] + t * (self.s[i + 1] - self.s[i])
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        ey = (x - proj[0]) * nx + (y - proj[1]) * ny
        return float(s_val), float(ey)

    def se2xy(self, s: float, ey: float):
        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s) - 2))
        r = (s - self.s[i]) / max(1e-9, self.s[i + 1] - self.s[i])
        base = self.P[i] * (1 - r) + self.P[i + 1] * r
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        x = base[0] + ey * nx
        y = base[1] + ey * ny
        return float(x), float(y)


# ====== 3) 生成 EGO：若基于锥桶失败则兜底到地图spawn点 ======
def spawn_ego_upstream_lane_center(env: HighwayEnv) -> carla.Actor:
    """
    【带诊断信息的版本】
    在第一个锥桶后方生成EGO，并打印详细的执行步骤。
    """
    print("\n--- [EGO 生成诊断 START] ---")
    world = env.world
    amap = world.get_map()
    ego_bp = get_ego_blueprint(world)

    # 第一步：尝试获取第一个锥桶的位置
    first_tf = env.get_first_cone_transform()

    if first_tf is not None:
        print(f"1. 成功获取到第一个锥桶的位置: {first_tf.location}")

        # 第二步：尝试在锥桶位置找到一个可行驶车道(Driving Lane)的路点
        wp = amap.get_waypoint(first_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)

        if wp is not None:
            print(f"2. 成功在锥桶位置附近找到可行驶车道的路点: {wp.transform.location}")

            # 第三步：循环尝试在路点后方不同距离生成车辆
            for back in [33.0, 34.0, 35.0]:  # 使用完整的距离列表以提高成功率
                print(f"3. 尝试在路点后方 {back}米 处寻找生成点...")
                prevs = wp.previous(back)

                if prevs:
                    spawn_wp = prevs[0]
                    print(f"   - 找到候选路点: {spawn_wp.transform.location}")

                    tf = carla.Transform(
                        carla.Location(
                            x=spawn_wp.transform.location.x,
                            y=spawn_wp.transform.location.y,
                            z=spawn_wp.transform.location.z + 1.0
                        ),
                        carla.Rotation(yaw=float(spawn_wp.transform.rotation.yaw))
                    )
                    tf_location = carla.Location(
                        x=spawn_wp.transform.location.x,
                        y=spawn_wp.transform.location.y,
                        z=spawn_wp.transform.location.z + 1.0
                    )
                    # 尝试生成
                    ego = world.try_spawn_actor(ego_bp, tf)
                    if ego:
                        env.set_ego(ego)
                        print(f"   ✅ [成功] 车辆已在后方 {back}米 处创建！")
                        print("--- [EGO 生成诊断 END] ---\n")
                        return ego , amap.get_waypoint(tf_location)
                    else:
                        print(f"   ❌ [失败] 生成失败。该位置可能被占用或无效。")
                else:
                    print(f"   - [跳过] 未能找到后方 {back}米 处的路点（可能道路太短）。")
        else:
            print("2. ❌ [失败] 在锥桶位置附近未能找到可行驶车道的路点。")
    else:
        print("1. ❌ [失败] 未能获取到第一个锥桶的位置。可能是场景生成失败。")

    # 如果以上所有步骤都失败了，启动后备方案
    print("\n[后备方案] 首选方案失败，现在尝试使用地图默认生成点...")
    spawns = amap.get_spawn_points()
    random.shuffle(spawns)
    for i, tf in enumerate(spawns[:10]):
        print(f"[后备方案] 尝试默认点 #{i + 1}...")
        tf.location.z += 0.20
        ego = world.try_spawn_actor(ego_bp, tf)
        if ego:
            env.set_ego(ego)
            print(f"   ✅ [成功] 车辆已在默认点创建！")
            print("--- [EGO 生成诊断 END] ---\n")
            return ego , None

    print("--- [EGO 生成诊断 END] ---\n")
    # 如果所有方案都失败，抛出最终错误
    raise RuntimeError("所有方案都已尝试，未能生成EGO。请检查上面的诊断日志确定失败环节。")


# ====== 4) 可行驶区域（走廊）——极简实现：地图车道线 + 障碍收紧 ======
def lane_bounds_from_map(world: carla.World, ref: LaneRef, s_arr: np.ndarray, lane_margin: float = 0.20):
    """
    对每个 s_i：以地图 Driving 车道宽得到左右边界（ey，左正右负）。
    返回 (left[], right[])。
    """
    amap = world.get_map()
    left = np.zeros_like(s_arr, dtype=float)
    right = np.zeros_like(s_arr, dtype=float)

    for i, s in enumerate(s_arr):
        x, y = ref.se2xy(float(s), 0.0)
        wp = amap.get_waypoint(carla.Location(x=x, y=y, z=0.0),
                               project_to_road=True,
                               lane_type=carla.LaneType.Driving)
        if wp is None:
            w = 3.5
        else:
            w = float(getattr(wp, "lane_width", 3.5)) or 3.5
        half = 0.5 * w
        left[i] = +half - lane_margin
        right[i] = -half + lane_margin
    return left, right


def shrink_by_obstacles(world: carla.World, ego: carla.Actor, ref: LaneRef,
                        s_arr: np.ndarray, left: np.ndarray, right: np.ndarray,
                        r_xy: float = 35.0, s_fwd: float = 20.0,
                        horizon_T: float = 1.5, dt: float = 0.2, obs_margin: float = 0.30):
    """
    在 s ∈ [s0, s0 + s_fwd] 内，用障碍“收紧”边界：
      - 动态：中心点常速外推
      - 静态：底面四角（若有bb）+ 中心点
    """
    ego_tf = ego.get_transform()
    s0, _ = ref.xy2se(ego_tf.location.x, ego_tf.location.y)
    s_min = s0
    s_max = s0 + float(s_fwd)

    s_start = float(s_arr[0])
    ds = float(s_arr[1] - s_arr[0]) if len(s_arr) >= 2 else 1.0

    def s_to_idx(s):
        return int(np.clip(round((s - s_start) / max(1e-6, ds)), 0, len(s_arr) - 1))

    ego_loc = ego_tf.location
    actors = world.get_actors()
    steps = max(1, int(horizon_T / max(1e-6, dt)))

    def consider_xy(x, y):
        try:
            s, ey = ref.xy2se(x, y)
        except Exception:
            return
        if not (s_min <= s <= s_max):
            return
        k = s_to_idx(s)
        # 只收紧车道内
        if ey >= 0.0:
            left[k] = min(left[k], ey - obs_margin)
        else:
            right[k] = max(right[k], ey + obs_margin)

    for a in actors:
        # 跳过自车/观众/hero
        try:
            if a.id == ego.id:
                continue
        except Exception:
            continue
        role = getattr(a, "attributes", {}).get("role_name", "")
        if role in ("hero", "spectator", "ego"):
            continue

        # 距离预筛
        try:
            loc = a.get_transform().location
        except Exception:
            continue
        dx, dy = loc.x - ego_loc.x, loc.y - ego_loc.y
        if dx * dx + dy * dy > r_xy * r_xy:
            continue

        tid = getattr(a, "type_id", "").lower()
        is_dyn = tid.startswith("vehicle.") or tid.startswith("walker.")

        if is_dyn:
            vel = a.get_velocity()
            for k in range(steps + 1):
                t = k * dt
                consider_xy(loc.x + vel.x * t, loc.y + vel.y * t)
        else:
            bb = getattr(a, "bounding_box", None)
            if bb is not None:
                try:
                    verts = bb.get_world_vertices(a.get_transform())
                    verts = sorted(verts, key=lambda v: v.z)[:4]  # 底面四角
                    for v in verts:
                        consider_xy(v.x, v.y)
                except Exception:
                    consider_xy(loc.x, loc.y)
            else:
                consider_xy(loc.x, loc.y)

    # 最小宽度保护 + 轻量平滑
    min_w = 1.8
    for i in range(len(s_arr)):
        if left[i] - right[i] < min_w:
            mid = 0.5 * (left[i] + right[i])
            left[i] = mid + 0.5 * min_w
            right[i] = mid - 0.5 * min_w
    if len(s_arr) >= 3:
        l2 = left.copy();
        r2 = right.copy()
        l2[1:-1] = (left[:-2] + 2.0 * left[1:-1] + left[2:]) * 0.25
        r2[1:-1] = (right[:-2] + 2.0 * right[1:-1] + right[2:]) * 0.25
        left[:] = 0.5 * left + 0.5 * l2
        right[:] = 0.5 * right + 0.5 * r2


# ====== 5) 规则型 Planner（极简：中线跟随 + 边界保护 + 简易速度）======
class RuleBasedPlanner:
    def __init__(self, ref: LaneRef, v_ref_base: float = 12.0):
        self.ref = ref
        self.v_ref_base = float(v_ref_base)
        self.corridor = None
        self._prev_delta = 0.0
        self._prev_ax = 0.0

    def update_corridor_simplified(self, world, ego, s_ahead=20.0, step=1.0, debug_draw=True):
        """
        动态生成前方走廊 (V2 - 修正版)。
        - 修正了障碍物边界计算错误的问题，确保走廊能正确避开障碍物。
        - 增大了障碍物的安全边距，使生成的路径更平滑、安全。
        """
        import carla
        import numpy as np

        if not ego:
            self.corridor = None
            return

        amap = world.get_map()
        dbg = world.debug

        ego_width = ego.bounding_box.extent.y * 2.0 + 0.4

        # ========== 1. 生成参考中心线 (不变) ==========
        center_waypoints = []
        current_wp = amap.get_waypoint(ego.get_location(), lane_type=carla.LaneType.Driving)
        center_waypoints.append(current_wp)
        for s in np.arange(step, s_ahead + step, step):
            next_wps = current_wp.next(s)
            if not next_wps: break
            current_wp = next_wps[0]  # 这里也需要更新
            center_waypoints.append(next_wps[0])

        if len(center_waypoints) < 2:
            self.corridor = None
            return

        # ========== 2. 核心计算逻辑 (已重构和修正) ==========
        def calculate_corridor_boundaries(waypoints, available_left_lane, available_right_lane):
            left_pts, right_pts = [], []
            cone_actors = world.get_actors().filter('static.prop.trafficcone*')

            # --- 修正点 1: 增大障碍物安全边距，避免锯齿 ---
            # 膨胀安全距离，从 0.8 扩大到 1.0
            OBSTACLE_MARGIN = 1.0
            LANE_MARGIN = 0.2

            for i, wp in enumerate(waypoints):
                wp_loc = wp.transform.location
                right_vec = wp.transform.get_right_vector()

                # --- a) 确定地图提供的最大可用宽度 ---
                max_left_width = wp.lane_width * 0.5 - LANE_MARGIN
                max_right_width = wp.lane_width * 0.5 - LANE_MARGIN

                if available_left_lane:
                    left_lane_wp = wp.get_left_lane()
                    if left_lane_wp: max_left_width += left_lane_wp.lane_width
                if available_right_lane:
                    right_lane_wp = wp.get_right_lane()
                    if right_lane_wp: max_right_width += right_lane_wp.lane_width

                # --- b) 应用障碍物约束 ---
                final_left_width = max_left_width
                final_right_width = max_right_width

                forward_vec = wp.transform.get_forward_vector()
                for cone in cone_actors:
                    vec_to_cone = cone.get_location() - wp_loc
                    if abs(vec_to_cone.dot(forward_vec)) > step * 1.5: continue

                    dist_lateral = vec_to_cone.dot(right_vec)

                    if dist_lateral < 0:
                        cone_limit_width = abs(dist_lateral) - OBSTACLE_MARGIN
                        final_left_width = min(final_left_width, cone_limit_width)
                    else:
                        cone_limit_width = dist_lateral - OBSTACLE_MARGIN
                        final_right_width = min(final_right_width, cone_limit_width)

                final_left_width = max(0, final_left_width)
                final_right_width = max(0, final_right_width)

                # --- c) 根据最终宽度计算边界点 ---
                left_offset = right_vec * -final_left_width
                left_pts.append(wp_loc + carla.Location(x=left_offset.x, y=left_offset.y, z=left_offset.z))

                right_offset = right_vec * final_right_width
                right_pts.append(wp_loc + carla.Location(x=right_offset.x, y=right_offset.y, z=right_offset.z))

            return left_pts, right_pts

        default_left_pts, default_right_pts = calculate_corridor_boundaries(center_waypoints, False, False)

        is_path_blocked = False
        for i in range(len(default_left_pts)):
            width = default_left_pts[i].distance(default_right_pts[i])
            if width < ego_width:
                is_path_blocked = True
                break

        final_left_pts, final_right_pts = default_left_pts, default_right_pts

        if is_path_blocked:
            can_go_left = False
            left_lane = current_wp.get_left_lane()
            if left_lane and left_lane.lane_type == carla.LaneType.Driving and str(current_wp.lane_change) in ("Left",
                                                                                                               "Both"):
                can_go_left = True

            can_go_right = False
            right_lane = current_wp.get_right_lane()
            if right_lane and right_lane.lane_type == carla.LaneType.Driving and str(current_wp.lane_change) in (
                    "Right", "Both"):
                can_go_right = True

            if can_go_left or can_go_right:
                final_left_pts, final_right_pts = calculate_corridor_boundaries(center_waypoints, can_go_left,
                                                                                can_go_right)

        # ========== 4. 可视化 (不变) ==========
        if debug_draw:
            z = ego.get_location().z + 0.2
            life_time = 0.1
            for i in range(len(final_left_pts) - 1):
                width = final_left_pts[i].distance(final_right_pts[i])
                color_left = carla.Color(255, 0, 0) if width < ego_width else carla.Color(64, 255, 255)
                color_right = carla.Color(255, 0, 0) if width < ego_width else carla.Color(255, 235, 64)

                p_left_1 = carla.Location(final_left_pts[i].x, final_left_pts[i].y, z)
                p_left_2 = carla.Location(final_left_pts[i + 1].x, final_left_pts[i + 1].y, z)
                p_right_1 = carla.Location(final_right_pts[i].x, final_right_pts[i].y, z)
                p_right_2 = carla.Location(final_right_pts[i + 1].x, final_right_pts[i + 1].y, z)

                # 正在测试场景 先不用花
                # dbg.draw_line(p_left_1, p_left_2, thickness=0.1, color=color_left, life_time=life_time,
                #               persistent_lines=False)
                # dbg.draw_line(p_right_1, p_right_2, thickness=0.1, color=color_right, life_time=life_time,
                #               persistent_lines=False)

        # ========== 5) 输出走廊到self.corridor ==========
        self.corridor = SimpleNamespace(
            s=np.linspace(0, s_ahead, len(final_left_pts), dtype=float),
            lower=np.array([p.y for p in final_right_pts], dtype=float),
            upper=np.array([p.y for p in final_left_pts], dtype=float)
        )

    def compute_control(self, ego: carla.Actor, dt: float = 0.05):
        """
        横向：纯追踪控制，取走廊中线作为目标路径
        纵向：简单的P控制器，以基础参考速度为目标
        """
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = float(math.hypot(vel.x, vel.y))
        x, y = tf.location.x, tf.location.y
        yaw_rad = math.radians(tf.rotation.yaw)

        # 车辆轴距 (一个合理的默认值)
        L = ego.bounding_box.extent.x * 2.0

        # 创建一个字典来存储调试信息，避免main函数中引用不存在的key
        dbg_info = {
            's': 0.0, 'ey': 0.0, 'lo': 0.0, 'up': 0.0, 'width': 0.0,
            'v': speed, 'v_ref': self.v_ref_base, 'delta': 0.0,
            'steer': 0.0, 'throttle': 0.0, 'brake': 0.0
        }

        # ========== 1. 检查走廊是否存在 ==========
        if self.corridor is None or len(self.corridor.s) < 2:
            # 没有走廊信息，直接刹车
            dbg_info['brake'] = 1.0
            return 0.0, 0.0, 1.0, dbg_info

        # ========== 2. 横向控制：纯追踪 (Pure Pursuit) ==========

        # a) 确定目标路径 (走廊中线)
        s_mid = self.corridor.s
        ey_mid = (self.corridor.upper + self.corridor.lower) / 2.0

        # b) 计算前视距离 L_d
        k_ld = 0.5  # 前视距离系数
        L_d = 5.0 + speed * k_ld

        # c) 找到目标点
        s_now, ey_now = self.ref.xy2se(x, y)
        s_target = s_now + L_d

        # 使用插值法找到目标点在走廊中线上的横向偏移ey
        ey_target = float(np.interp(np.clip(s_target, s_mid[0], s_mid[-1]), s_mid, ey_mid))

        # d) 将目标点从Frenet坐标系转换到世界坐标系
        target_x, target_y = self.ref.se2xy(s_target, ey_target)

        # e) 计算目标点相对于车头方向的角度 alpha
        vec_to_target_x = target_x - x
        vec_to_target_y = target_y - y
        alpha = math.atan2(vec_to_target_y, vec_to_target_x) - yaw_rad

        # f) 计算转向角 delta
        delta = math.atan2(2.0 * L * math.sin(alpha), L_d)

        # g) 将转向角转换为 [-1, 1] 的steer值
        # 假设最大方向盘转角为30度
        max_steer_angle = math.radians(30.0)
        steer = float(np.clip(delta / max_steer_angle, -1.0, 1.0))

        # ========== 3. 纵向控制：简单P控制器 ==========
        v_ref = self.v_ref_base

        # 简单规则：如果前方走廊很窄，就减速
        # 找到车辆前方约3-8米处的走廊宽度
        s_check_start = s_now + 3.0
        s_check_end = s_now + 8.0

        indices = np.where((s_mid >= s_check_start) & (s_mid <= s_check_end))
        if len(indices[0]) > 0:
            width_ahead = self.corridor.upper[indices] - self.corridor.lower[indices]
            min_width_ahead = np.min(width_ahead)
            if min_width_ahead < L + 0.8:  # 如果宽度小于车宽+0.8米
                v_ref *= 0.5  # 目标速度减半

        # P控制器
        error = v_ref - speed
        kp = 0.4  # 比例系数
        throttle = kp * error

        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = 0.0
        if error < -0.5:  # 如果速度超出目标速度较多，则刹车
            throttle = 0.0
            brake = float(np.clip(-error * 0.5, 0.0, 1.0))

        # ========== 4. 更新调试信息并返回 ==========
        s_idx = np.argmin(np.abs(self.corridor.s - s_now))
        dbg_info.update({
            's': s_now, 'ey': ey_now,
            'lo': self.corridor.lower[s_idx], 'up': self.corridor.upper[s_idx],
            'width': self.corridor.upper[s_idx] - self.corridor.lower[s_idx],
            'v_ref': v_ref, 'delta': delta, 'steer': steer,
            'throttle': throttle, 'brake': brake
        })
        # 仅场景测试使用
        throttle =0
        brake =1
        return throttle, steer, brake, dbg_info




# ====== 6) 主程序 ======
def main():
    env = HighwayEnv(host="127.0.0.1", port=2000, sync=True, fixed_dt=0.05).connect()
    logger = None
    try:
        env.setup_scene(
            num_cones=10, step_forward=3.0, step_right=0.35,
            z_offset=0.0, min_gap_from_junction=15.0,
            grid=5.0, set_spectator=True
        )

        ego , ego_wp= spawn_ego_upstream_lane_center(env)
        idp = 0.8  #  这里切换周围交通参与者的密度
        scenemanager = SceneManager(ego_wp, idp)
        scenemanager.gen_traffic_flow(env.world, ego_wp)

        # 参考线：用第一个锥桶所在驾驶车道作为种子
        amap = env.world.get_map()
        first_tf = env.get_first_cone_transform()
        if first_tf is None:
            # 兜底：用自车位置投影到Driving车道
            seed_wp = amap.get_waypoint(ego.get_transform().location,
                                        project_to_road=True, lane_type=carla.LaneType.Driving)
        else:
            seed_wp = amap.get_waypoint(first_tf.location,
                                        project_to_road=True, lane_type=carla.LaneType.Driving)
        ref = LaneRef(amap, seed_wp=seed_wp, step=1.0, max_len=500.0)

        planner = RuleBasedPlanner(ref, v_ref_base=12.0)
        logger = TelemetryLogger(out_dir="logs_rule_based")

        dt = 0.05
        frame = 0

        # import ipdb
        # ipdb.set_trace()
        while True:
            # 1) 更新走廊（始终有线；有障碍则收紧）
            planner.update_corridor_simplified(env.world, ego, s_ahead=20.0, step=1.0)
            draw_corridor(env.world, ref, planner.corridor)

            # 2) 控制
            throttle, steer, brake, dbg = planner.compute_control(ego, dt=dt)

            # 3) 执行
            env.apply_control(throttle=throttle, steer=steer, brake=brake)

            # 4) 仿真步进 & 可视化
            obs, _ = env.step()
            if frame % 2 == 0:
                tf = ego.get_transform()
                draw_ego_marker(env.world, tf.location.x, tf.location.y)

            # 5) 打印 & 记录
            if frame % 10 == 0:
                print(f"[CTRL] s={dbg['s']:.1f} ey={dbg['ey']:.2f} | lo={dbg['lo']:.2f} up={dbg['up']:.2f} "
                      f"w={dbg['width']:.2f} | v={dbg['v']:.2f}->{dbg['v_ref']:.2f} "
                      f"| delta={dbg['delta']:.3f} steer={dbg['steer']:.2f}")
                print(f"[LONG] th={dbg['throttle']:.2f} br={dbg['brake']:.2f}")

            logger.log(frame, obs, dbg, ref)
            frame += 1

    except KeyboardInterrupt:
        print("\n[Stop] 手动退出。")
    finally:
        try:
            if logger is not None:
                logger.save_csv()
                logger.plot()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()