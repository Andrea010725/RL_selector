# agents/rule_based/agent.py
from __future__ import annotations
import math
import random
import sys
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Any, List

from scipy.optimize import minimize

sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np

# 你的工程依赖
sys.path.append("/home/ajifang/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint

# ✅ 导入场景类
from env.scenarios import JaywalkerScenario, TrimmaScenario, ConstructionLaneChangeScenario, ConesScenario

# 交通流工具（如你不需要，可以不启用；但保留不影响）
sys.path.append("/home/ajifang/Driveadapter_2/tools")
from custom_eval import TrafficFlowSpawner

from vis_debug import TelemetryLogger


# ============================================================
# 颜色工具
# ============================================================
def _col(r, g, b):
    return carla.Color(int(r), int(g), int(b))

COL_REF   = _col(200, 200, 200)   # 灰：参考线
COL_LEFT  = _col(255, 0, 255)     # 紫：物理左边界
COL_RIGHT = _col(50, 220, 50)     # 绿：物理右边界
COL_DP    = _col(255, 255, 0)     # 黄：DP/NMPC跟踪中心线


# ============================================================
# 1) LaneRef：局部参考线 + Frenet变换 + 曲率（用于 yaw_err 动力学）
# ============================================================
class LaneRef:
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len: float = 200.0):
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
        if len(self.P) < 2:
            self.s = np.array([0.0], dtype=float)
            self.tang = np.array([[1.0, 0.0]], dtype=float)
        else:
            d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
            self.s = np.concatenate([[0.0], np.cumsum(d)])  # [N]
            tang = np.diff(self.P, axis=0)
            tang = np.vstack([tang, tang[-1]])
            self.tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)

        self.wps = wps
        self.step = float(step)

        # 曲率 kappa(s)
        if len(self.P) < 3:
            self.kappa = np.zeros(len(self.s), dtype=float)
        else:
            psi = np.arctan2(self.tang[:, 1], self.tang[:, 0])  # [N]
            dpsi = np.diff(psi)
            dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
            ds_arr = np.diff(self.s) + 1e-9
            k_mid = dpsi / ds_arr  # [N-1]

            self.kappa = np.zeros(len(self.s), dtype=float)
            if len(self.s) >= 3:
                self.kappa[1:-1] = 0.5 * (k_mid[:-1] + k_mid[1:])
            self.kappa[0] = k_mid[0]
            self.kappa[-1] = k_mid[-1]

    def kappa_at_s(self, s: float) -> float:
        if len(self.s) < 2:
            return 0.0
        s = float(np.clip(s, self.s[0], self.s[-1]))
        return float(np.interp(s, self.s, self.kappa))

    def _segment_index_and_t(self, x, y) -> Tuple[int, float, np.ndarray, float]:
        P = self.P
        xy = np.array([x, y], dtype=float)

        if len(P) < 2:
            proj = P[0].copy()
            dist = float(np.linalg.norm(xy - proj))
            return 0, 0.0, proj, dist

        v = xy - P[:-1]             # [N-1,2]
        seg = P[1:] - P[:-1]        # [N-1,2]
        seg_len2 = (seg[:, 0] ** 2 + seg[:, 1] ** 2) + 1e-9

        t = np.clip((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:, None]
        dist2 = np.sum((proj - xy[None, :]) ** 2, axis=1)

        i = int(np.argmin(dist2))
        min_dist = float(math.sqrt(dist2[i]))
        return i, float(t[i]), proj[i], min_dist

    def xy2se(self, x: float, y: float, max_proj_dist: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
        i, t, proj, min_dist = self._segment_index_and_t(x, y)
        if (max_proj_dist is not None) and (min_dist > max_proj_dist):
            return None, None

        if len(self.s) < 2:
            s_val = 0.0
        else:
            s_val = self.s[i] + t * (self.s[i + 1] - self.s[i])

        tx, ty = self.tang[min(i, len(self.tang) - 1)]
        nx, ny = -ty, tx
        ey = (x - proj[0]) * nx + (y - proj[1]) * ny
        return float(s_val), float(ey)

    def se2xy(self, s: float, ey: float) -> Tuple[float, float]:
        if len(self.P) < 2:
            return float(self.P[0, 0]), float(self.P[0, 1])

        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s) - 2))

        ds = max(1e-9, self.s[i + 1] - self.s[i])
        r = (s - self.s[i]) / ds

        base = self.P[i] * (1 - r) + self.P[i + 1] * r
        tx, ty = self.tang[i]
        nx, ny = -ty, tx

        x = base[0] + ey * nx
        y = base[1] + ey * ny
        return float(x), float(y)


# ============================================================
# 2) spawn ego - 支持场景和锥桶两种模式
# ============================================================
def spawn_ego_from_scenario(world: carla.World, scenario, env: Optional[HighwayEnv] = None) -> Tuple[carla.Actor, carla.Waypoint]:
    print("\n--- [EGO 场景生成 START] ---")
    amap = world.get_map()
    ego_bp = get_ego_blueprint(world)

    spawn_tf = scenario.get_spawn_transform()
    if spawn_tf is None:
        raise RuntimeError("场景未能提供有效的 spawn transform")

    print(f"1. 场景提供的生成位置: ({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f}, {spawn_tf.location.z:.1f})")

    ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        spawn_tf.location.z += 0.5
        ego = world.try_spawn_actor(ego_bp, spawn_tf)

    if ego is None:
        raise RuntimeError(f"无法在场景提供的位置生成 ego: {spawn_tf.location}")

    if env is not None:
        env.set_ego(ego)

    ego_wp = amap.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
    if ego_wp is None:
        raise RuntimeError("Ego 生成成功，但无法获取 ego_wp（地图投影失败）")

    print("2. ✅ Ego 成功生成在场景位置")
    print("--- [EGO 场景生成 END] ---\n")
    return ego, ego_wp


def spawn_ego_upstream_lane_center(env: HighwayEnv) -> Tuple[carla.Actor, carla.Waypoint]:
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
                    tf = carla.Transform(
                        carla.Location(
                            x=spawn_wp.transform.location.x,
                            y=spawn_wp.transform.location.y,
                            z=spawn_wp.transform.location.z + 1.0
                        ),
                        carla.Rotation(yaw=float(spawn_wp.transform.rotation.yaw))
                    )
                    ego = world.try_spawn_actor(ego_bp, tf)
                    if ego:
                        env.set_ego(ego)
                        ego_wp = amap.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
                        if ego_wp is None:
                            raise RuntimeError("Ego 生成成功，但无法获取 ego_wp（地图投影失败）")
                        print(f"    ✅ [成功] 车辆已在后方 {back}米 处创建！")
                        print("--- [EGO 生成诊断 END] ---\n")
                        return ego, ego_wp

    print("\n[后备方案] 首选方案失败，现在尝试使用地图默认生成点...")
    spawns = amap.get_spawn_points()
    random.shuffle(spawns)

    for i, tf in enumerate(spawns[:10]):
        tf.location.z += 0.20
        ego = world.try_spawn_actor(ego_bp, tf)
        if ego:
            env.set_ego(ego)
            ego_wp = amap.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            if ego_wp is None:
                raise RuntimeError("Ego 默认点生成成功，但无法获取 ego_wp")
            print(f"    ✅ [成功] 车辆已在默认点创建！")
            print("--- [EGO 生成诊断 END] ---\n")
            return ego, ego_wp

    print("--- [EGO 生成诊断 END] ---\n")
    raise RuntimeError("所有方案都已尝试，未能生成EGO。")


def spectator_follow_ego(world: carla.World, ego: carla.Actor,
                         height: float = 8.0, distance_behind: float = 6.0,
                         pitch: float = -20.0):
    if ego is None:
        return
    tf = ego.get_transform()
    forward = tf.get_forward_vector()
    cam_loc = tf.location - forward * distance_behind + carla.Location(z=height)
    cam_rot = carla.Rotation(pitch=pitch, yaw=tf.rotation.yaw, roll=0.0)
    world.get_spectator().set_transform(carla.Transform(cam_loc, cam_rot))


# ============================================================
# 5) Corridor：严格 one-sided cones + 连续 cone 边界 + 强制锥桶安全距离
# ============================================================
def build_corridor_by_cones_one_side_only(
        world: carla.World,
        ego: carla.Actor,
        ref: LaneRef,
        s_ahead: float = 30.0,
        ds: float = 1.0,
        lane_margin: float = 0.20,
        cone_margin: float = 0.30,
        min_width: float = 1.8,
        cone_extra_clearance: float = 0.80,
        cone_s_ext_back: float = 0.0,
        cone_s_ext_front: float = 0.0,
        expand_adjacent: bool = False,
        required_width: Optional[float] = None,
):
    amap = world.get_map()

    ego_tf = ego.get_transform()
    s0, _ = ref.xy2se(ego_tf.location.x, ego_tf.location.y, max_proj_dist=None)
    if s0 is None:
        return None

    # ✅ 仅考虑“自车前方20m、同车道”的交通参与者
    ego_wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_road_id = ego_wp.road_id if ego_wp is not None else None
    ego_lane_id = ego_wp.lane_id if ego_wp is not None else None
    front_consider_dist = 20.0

    s_nodes = np.arange(float(s0), float(s0) + float(s_ahead), float(ds))

    def _lane_bound_points(wp: carla.Waypoint):
        c = wp.transform.location
        right_vec = wp.transform.get_right_vector()
        w = float(getattr(wp, "lane_width", 3.5)) or 3.5
        half = 0.5 * w
        left_edge = carla.Location(x=c.x - right_vec.x * half, y=c.y - right_vec.y * half, z=c.z)
        right_edge = carla.Location(x=c.x + right_vec.x * half, y=c.y + right_vec.y * half, z=c.z)
        return left_edge, right_edge

    # 1) 外侧边界
    left_outer_ey = np.zeros(len(s_nodes), dtype=float)
    right_outer_ey = np.zeros(len(s_nodes), dtype=float)
    left_outer_world = []
    right_outer_world = []

    for i, s in enumerate(s_nodes):
        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_center is None:
            left_outer_ey[i] = +1.75 - lane_margin
            right_outer_ey[i] = -1.75 + lane_margin
            left_outer_world.append(carla.Location(x=cx, y=cy, z=0.0))
            right_outer_world.append(carla.Location(x=cx, y=cy, z=0.0))
            continue

        ego_left_edge, ego_right_edge = _lane_bound_points(wp_center)

        if not expand_adjacent:
            left_outer_pt = ego_left_edge
            right_outer_pt = ego_right_edge
        else:
            wp_left = wp_center.get_left_lane()
            wp_right = wp_center.get_right_lane()

            if (wp_left is not None) and (wp_left.lane_type == carla.LaneType.Driving):
                left_edge, _ = _lane_bound_points(wp_left)
                left_outer_pt = left_edge
            else:
                left_outer_pt = ego_left_edge

            if (wp_right is not None) and (wp_right.lane_type == carla.LaneType.Driving):
                _, right_edge = _lane_bound_points(wp_right)
                right_outer_pt = right_edge
            else:
                right_outer_pt = ego_right_edge

        right_vec = wp_center.transform.get_right_vector()

        left_outer_pt_in = carla.Location(
            x=left_outer_pt.x + right_vec.x * lane_margin,
            y=left_outer_pt.y + right_vec.y * lane_margin,
            z=left_outer_pt.z
        )
        right_outer_pt_in = carla.Location(
            x=right_outer_pt.x - right_vec.x * lane_margin,
            y=right_outer_pt.y - right_vec.y * lane_margin,
            z=right_outer_pt.z
        )

        _, ey_L_in = ref.xy2se(left_outer_pt_in.x, left_outer_pt_in.y, max_proj_dist=None)
        _, ey_R_in = ref.xy2se(right_outer_pt_in.x, right_outer_pt_in.y, max_proj_dist=None)

        left_outer_ey[i] = float(ey_L_in)
        right_outer_ey[i] = float(ey_R_in)
        left_outer_world.append(left_outer_pt_in)
        right_outer_world.append(right_outer_pt_in)

    # 2) 收集锥桶
    cones = []
    for a in world.get_actors():
        if getattr(a, "id", None) == ego.id:
            continue
        tname = (getattr(a, "type_id", "") or "").lower()
        if not tname.startswith("static.prop."):
            continue
        if ("trafficcone" not in tname) and ("traffic_cone" not in tname):
            continue

        loc = a.get_location()
        wp = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue

        right_vec = wp.transform.get_right_vector()
        center = wp.transform.location
        dot = (loc.x - center.x) * right_vec.x + (loc.y - center.y) * right_vec.y
        side = "RIGHT" if dot > 0.0 else "LEFT"

        s_cone, ey_cone = ref.xy2se(loc.x, loc.y, max_proj_dist=None)
        if s_cone is None:
            continue
        cones.append((float(s_cone), float(ey_cone), side, loc))

    if len(cones) == 0:
        left_phys = left_outer_ey.copy()
        right_phys = right_outer_ey.copy()
        bound_upper = np.maximum(left_phys, right_phys)
        bound_lower = np.minimum(left_phys, right_phys)
        return dict(
            s_nodes=s_nodes,
            left_phys=left_phys,
            right_phys=right_phys,
            bound_upper=bound_upper,
            bound_lower=bound_lower,
            bound_upper_safe=bound_upper.copy(),
            bound_lower_safe=bound_lower.copy(),
            upper_pts_world=left_outer_world[:],
            lower_pts_world=right_outer_world[:],
            mode="NONE",
            cone_boundary_ey=np.full(len(s_nodes), np.nan, dtype=float),
            min_width_safe=float(np.min(bound_upper - bound_lower)),
            expanded=bool(expand_adjacent),
        )

    cnt_R = sum(1 for (_, _, side, _) in cones if side == "RIGHT")
    cnt_L = sum(1 for (_, _, side, _) in cones if side == "LEFT")

    if cnt_R > cnt_L:
        mode = "RIGHT"
    elif cnt_L > cnt_R:
        mode = "LEFT"
    else:
        cones_sorted = sorted(cones, key=lambda x: abs(x[0] - float(s0)))
        mode = cones_sorted[0][2]

    print(f"\n[Corridor-By-Cones] cones 侧判定：LEFT={cnt_L}, RIGHT={cnt_R} -> mode={mode}\n")

    # 3) 连续 cone 边界：按 s 插值
    cones_side = [(sc, ey, loc) for (sc, ey, side, loc) in cones if side == mode]
    cones_side.sort(key=lambda x: x[0])
    sc_list = np.array([c[0] for c in cones_side], dtype=float)
    ey_list = np.array([c[1] for c in cones_side], dtype=float)

    cone_boundary_ey = np.full(len(s_nodes), np.nan, dtype=float)
    if len(sc_list) >= 2:
        # ✅ 只在锥桶实际分布区间内插值，避免“锥桶约束”被无限延伸
        s_min, s_max = float(sc_list[0]), float(sc_list[-1])
        mask_core = (s_nodes >= s_min) & (s_nodes <= s_max)
        cone_boundary_ey[mask_core] = np.interp(s_nodes[mask_core], sc_list, ey_list)

        # ✅ 纵向膨胀：只在首尾锥桶前后延展（保持横向不变）
        s_min_ext = s_min - max(0.0, float(cone_s_ext_back))
        s_max_ext = s_max + max(0.0, float(cone_s_ext_front))
        mask_ext = (s_nodes >= s_min_ext) & (s_nodes <= s_max_ext)

        # 前后延展区用首尾锥桶的 ey 保持
        if np.any(s_nodes < s_min):
            mask_back = (s_nodes >= s_min_ext) & (s_nodes < s_min)
            cone_boundary_ey[mask_back] = float(ey_list[0])
        if np.any(s_nodes > s_max):
            mask_front = (s_nodes > s_max) & (s_nodes <= s_max_ext)
            cone_boundary_ey[mask_front] = float(ey_list[-1])
        mask = mask_ext
    else:
        # ✅ 单个锥桶：仅在其附近给出边界，避免影响远处
        s_only = float(sc_list[0])
        mask = np.abs(s_nodes - s_only) <= 2.0
        cone_boundary_ey[mask] = float(ey_list[0])

    def _smooth_nan_aware(arr, k=7):
        a = arr.copy()
        idx = np.where(np.isfinite(a))[0]
        if len(idx) < 3:
            return a
        filled = a.copy()
        last = None
        for i in range(len(filled)):
            if np.isfinite(filled[i]):
                last = filled[i]
            elif last is not None:
                filled[i] = last
        last = None
        for i in range(len(filled) - 1, -1, -1):
            if np.isfinite(filled[i]):
                last = filled[i]
            elif last is not None:
                filled[i] = last
        ker = np.ones(k, dtype=float) / float(k)
        sm = np.convolve(filled, ker, mode="same")
        out = a.copy()
        out[np.isfinite(a)] = sm[np.isfinite(a)]
        return out

    # ✅ 只在锥桶区间内平滑，区间外保持 NaN
    if "mask" in locals() and np.any(mask):
        smoothed = _smooth_nan_aware(cone_boundary_ey, k=7)
        cone_boundary_ey = np.where(mask, smoothed, np.nan)

    # 4) 把 cone 边界按物理方向推开 cone_margin
    for i, s in enumerate(s_nodes):
        ey = cone_boundary_ey[i]
        if not np.isfinite(ey):
            continue

        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_center is None:
            continue

        right_vec = wp_center.transform.get_right_vector()
        px, py = ref.se2xy(float(s), float(ey))

        if mode == "RIGHT":
            p = carla.Location(x=px + right_vec.x * cone_margin, y=py + right_vec.y * cone_margin, z=wp_center.transform.location.z)
        else:
            p = carla.Location(x=px - right_vec.x * cone_margin, y=py - right_vec.y * cone_margin, z=wp_center.transform.location.z)

        _, ey_in = ref.xy2se(p.x, p.y, max_proj_dist=None)
        cone_boundary_ey[i] = float(ey_in)

    # 5) 生成 corridor 两端 + 安全走廊
    endA_ey = np.zeros(len(s_nodes), dtype=float)
    endB_ey = np.zeros(len(s_nodes), dtype=float)
    endA_world, endB_world = [], []
    other_side_ey = np.zeros(len(s_nodes), dtype=float)

    for i, s in enumerate(s_nodes):
        cey = cone_boundary_ey[i]
        if not np.isfinite(cey):
            a_ey = left_outer_ey[i]
            b_ey = right_outer_ey[i]
            other_side_ey[i] = (right_outer_ey[i] if mode == "RIGHT" else left_outer_ey[i])
        else:
            if mode == "RIGHT":
                a_ey = cey
                b_ey = right_outer_ey[i]
                other_side_ey[i] = right_outer_ey[i]
            else:
                a_ey = left_outer_ey[i]
                b_ey = cey
                other_side_ey[i] = left_outer_ey[i]

        endA_ey[i] = float(a_ey)
        endB_ey[i] = float(b_ey)
        ax, ay = ref.se2xy(float(s), float(a_ey))
        bx, by = ref.se2xy(float(s), float(b_ey))
        endA_world.append(carla.Location(x=ax, y=ay, z=0.0))
        endB_world.append(carla.Location(x=bx, y=by, z=0.0))

    # 物理左/右（用于画图）
    left_phys = np.zeros(len(s_nodes), dtype=float)
    right_phys = np.zeros(len(s_nodes), dtype=float)
    left_pts_world, right_pts_world = [], []

    for i, s in enumerate(s_nodes):
        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp_center is None:
            left_phys[i] = endA_ey[i]
            right_phys[i] = endB_ey[i]
            left_pts_world.append(endA_world[i])
            right_pts_world.append(endB_world[i])
            continue

        right_vec = wp_center.transform.get_right_vector()
        center = wp_center.transform.location

        A = endA_world[i]
        B = endB_world[i]
        dotA = (A.x - center.x) * right_vec.x + (A.y - center.y) * right_vec.y
        dotB = (B.x - center.x) * right_vec.x + (B.y - center.y) * right_vec.y

        if dotA <= dotB:
            left_phys[i] = endA_ey[i]
            right_phys[i] = endB_ey[i]
            left_pts_world.append(A)
            right_pts_world.append(B)
        else:
            left_phys[i] = endB_ey[i]
            right_phys[i] = endA_ey[i]
            left_pts_world.append(B)
            right_pts_world.append(A)

    bound_upper = np.maximum(left_phys, right_phys)
    bound_lower = np.minimum(left_phys, right_phys)

    # 6) 强制锥桶安全距离
    ego_width = float(ego.bounding_box.extent.y * 2.0)
    cone_clear = float(cone_margin + 0.5 * ego_width + cone_extra_clearance)

    bound_upper_safe = bound_upper.copy()
    bound_lower_safe = bound_lower.copy()

    for i in range(len(s_nodes)):
        cey = cone_boundary_ey[i]
        oey = other_side_ey[i]
        if not np.isfinite(cey):
            continue
        if oey > cey:
            bound_lower_safe[i] = max(bound_lower_safe[i], cey + cone_clear)
        else:
            bound_upper_safe[i] = min(bound_upper_safe[i], cey - cone_clear)

        if bound_upper_safe[i] <= bound_lower_safe[i]:
            mid = 0.5 * (bound_upper[i] + bound_lower[i])
            bound_upper_safe[i] = mid + 0.3
            bound_lower_safe[i] = mid - 0.3

    # 车辆挤占（静态当前帧）
    # ✅ 车辆横向挤占：加入速度线性膨胀（使用 CARLA API 动态计算）
    car_clear_lat_base = 0.5 * ego_width + 0.30
    car_influence_s = 8.0
    for a in world.get_actors():
        if getattr(a, "id", None) == ego.id:
            continue
        tname = (getattr(a, "type_id", "") or "").lower()
        if not tname.startswith("vehicle."):
            continue
        loc = a.get_location()
        s_car, ey_car = ref.xy2se(loc.x, loc.y, max_proj_dist=None)
        if s_car is None:
            continue

        # ✅ 只考虑“同车道、且在自车前方20m内”的车辆
        if ego_wp is None:
            continue
        wp_car = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if (wp_car is None) or (wp_car.road_id != ego_road_id) or (wp_car.lane_id != ego_lane_id):
            continue
        if not (0.0 < (float(s_car) - float(s0)) <= front_consider_dist):
            continue
        # 速度线性膨胀（不使用固定 speed_k）
        try:
            vel = a.get_velocity()
            v = float(math.hypot(vel.x, vel.y))  # m/s
        except Exception:
            v = 0.0

        # 取该车所在车道宽与限速，构造线性膨胀（速度比例）
        lane_width = 3.5
        speed_limit_kmh = None
        try:
            wp = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                lane_width = float(getattr(wp, "lane_width", 3.5)) or 3.5
                if hasattr(wp, "get_speed_limit"):
                    speed_limit_kmh = float(wp.get_speed_limit())
        except Exception:
            pass
        try:
            if hasattr(a, "get_speed_limit"):
                speed_limit_kmh = float(a.get_speed_limit())
        except Exception:
            pass
        if speed_limit_kmh is None or speed_limit_kmh <= 1e-3:
            speed_limit_kmh = 50.0
        speed_limit_mps = float(speed_limit_kmh) / 3.6
        ratio = min(1.0, max(0.0, v / max(5.0, speed_limit_mps)))
        speed_extra = float(np.clip(lane_width * 0.35 * ratio, 0.0, 1.2))

        car_clear_lat = car_clear_lat_base + speed_extra
        for i, s in enumerate(s_nodes):
            if abs(float(s) - float(s_car)) > car_influence_s:
                continue
            if ey_car >= 0.0:
                bound_lower_safe[i] = max(bound_lower_safe[i], float(ey_car) + car_clear_lat)
            else:
                bound_upper_safe[i] = min(bound_upper_safe[i], float(ey_car) - car_clear_lat)

            if bound_upper_safe[i] <= bound_lower_safe[i]:
                mid = 0.5 * (bound_upper[i] + bound_lower[i])
                bound_upper_safe[i] = mid + 0.3
                bound_lower_safe[i] = mid - 0.3

    # 7) 最小宽度兜底
    width_thresh = max(float(min_width), float(ego_width + 0.20))
    for i in range(len(s_nodes)):
        w = float(bound_upper_safe[i] - bound_lower_safe[i])
        if w >= width_thresh:
            continue
        mid = 0.5 * (bound_upper_safe[i] + bound_lower_safe[i])
        bound_upper_safe[i] = mid + 0.5 * width_thresh
        bound_lower_safe[i] = mid - 0.5 * width_thresh

    min_width_safe = float(np.min(bound_upper_safe - bound_lower_safe))

    return dict(
        s_nodes=s_nodes,
        left_phys=left_phys,
        right_phys=right_phys,
        bound_upper=bound_upper,
        bound_lower=bound_lower,
        bound_upper_safe=bound_upper_safe,
        bound_lower_safe=bound_lower_safe,
        upper_pts_world=left_pts_world,
        lower_pts_world=right_pts_world,
        mode=mode,
        cone_boundary_ey=cone_boundary_ey,
        min_width_safe=min_width_safe,
        expanded=bool(expand_adjacent),
    )


# ============================================================
# 6) DP：生成 center_path_ey（黄色线）+ blocked_intervals(动态预测障碍物)
# ============================================================
def dp_plan_centerline(
    world: carla.World,
    ego: carla.Actor,
    ref: LaneRef,
    s_nodes: np.ndarray,
    bound_upper: np.ndarray,
    bound_lower: np.ndarray,
    *,
    ey_range: float = 6.0,
    dey: float = 0.15,
    corridor_margin: float = 0.20,
    W_CENTER: float = 2.0,
    W_SMOOTH: float = 12.0,
    fallback_to_mid: bool = True,
    blocked_intervals: Optional[List[List[Tuple[float, float]]]] = None,  # len==Ns, each=[(lo,up)...]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    INF = 1e18
    ey_grid = np.arange(-ey_range, ey_range + 1e-6, dey, dtype=float)
    Ny = len(ey_grid)
    Ns = len(s_nodes)

    ego_loc = ego.get_location()
    ego_wp = world.get_map().get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    lane_w = float(getattr(ego_wp, "lane_width", 3.5)) if ego_wp is not None else 3.5
    max_proj_dist = 1.2 * lane_w + 1.0

    s0, ey0 = ref.xy2se(ego_loc.x, ego_loc.y, max_proj_dist=max_proj_dist)
    if s0 is None:
        if fallback_to_mid:
            return 0.5 * (bound_upper + bound_lower), {"dp_ok": False, "reason": "ego_out_of_ref"}
        return np.zeros(Ns), {"dp_ok": False, "reason": "ego_out_of_ref"}

    j_start = int(np.argmin(np.abs(ey_grid - float(ey0))))
    cost = np.full((Ns, Ny), INF, dtype=float)

    for i in range(Ns):
        lo = float(bound_lower[i]) + corridor_margin
        up = float(bound_upper[i]) - corridor_margin
        if up <= lo:
            lo = float(bound_lower[i])
            up = float(bound_upper[i])

        valid = (ey_grid >= lo) & (ey_grid <= up)

        # 扣掉动态障碍物禁行区间
        if blocked_intervals is not None and i < len(blocked_intervals):
            for (blo, bup) in blocked_intervals[i]:
                valid &= ~((ey_grid >= float(blo)) & (ey_grid <= float(bup)))

        mid = 0.5 * (lo + up)

        if not np.any(valid):
            j_mid = int(np.argmin(np.abs(ey_grid - mid)))
            valid[:] = False
            valid[j_mid] = True

        cost[i, valid] = W_CENTER * ((ey_grid[valid] - mid) ** 2)

    if not np.isfinite(cost[0, j_start]) or cost[0, j_start] >= INF * 0.5:
        if fallback_to_mid:
            return 0.5 * (bound_upper + bound_lower), {"dp_ok": False, "reason": "start_infeasible"}
        return np.zeros(Ns), {"dp_ok": False, "reason": "start_infeasible"}

    dp = np.full((Ns, Ny), INF, dtype=float)
    prev = np.full((Ns, Ny), -1, dtype=int)
    dp[0, j_start] = cost[0, j_start]

    K = 3
    for i in range(1, Ns):
        for j in range(Ny):
            if cost[i, j] >= INF * 0.5:
                continue
            j_lo = max(0, j - K)
            j_hi = min(Ny - 1, j + K)
            best, bestp = INF, -1
            for jp in range(j_lo, j_hi + 1):
                if dp[i - 1, jp] >= INF * 0.5:
                    continue
                dy = ey_grid[j] - ey_grid[jp]
                v = dp[i - 1, jp] + cost[i, j] + W_SMOOTH * (dy * dy)
                if v < best:
                    best, bestp = v, jp
            dp[i, j] = best
            prev[i, j] = bestp

    j_end = int(np.argmin(dp[-1]))
    if not np.isfinite(dp[-1, j_end]) or dp[-1, j_end] >= INF * 0.5:
        if fallback_to_mid:
            return 0.5 * (bound_upper + bound_lower), {"dp_ok": False, "reason": "dp_inf"}
        return np.zeros(Ns), {"dp_ok": False, "reason": "dp_inf"}

    j_path = np.zeros(Ns, dtype=int)
    j_path[-1] = j_end
    for i in range(Ns - 1, 0, -1):
        j_path[i - 1] = prev[i, j_path[i]]
        if j_path[i - 1] < 0:
            if fallback_to_mid:
                return 0.5 * (bound_upper + bound_lower), {"dp_ok": False, "reason": "backtrack_fail"}
            break

    ey_ref = ey_grid[j_path].astype(float)
    return ey_ref, {"dp_ok": True, "min_cost": float(dp[-1, j_end])}


# ============================================================
# 7) RuleBasedPlanner：动态参考线 + corridor + DP + 软约束NMPC
# ============================================================
class RuleBasedPlanner:
    """
    ✅ 关键恢复点：
    - Planner 持有 amap
    - 每次 update_corridor 都从 ego 重建参考线 ref（避免 ego_out_of_ref 一启动就退出）
    - 保留 blocked_intervals（动态障碍物预测 tube）
    """
    def __init__(self, amap: carla.Map, v_ref_base: float = 12.0):
        self.amap = amap
        self.v_ref_base = float(v_ref_base)
        self.ref: Optional[LaneRef] = None
        self.corridor = None

        self.DRAW_REF_LINE = True
        self.DRAW_CORRIDOR_EDGES = True
        self.DRAW_DP_CENTER = True

        self.DRAW_PERIOD = 0.10
        self.DEBUG_LIFE_TIME = 0.12
        self._last_draw_t = -1e9

        self.DP_ENABLE = True
        self.DP_EY_RANGE = 6.0
        self.DP_DEY = 0.15
        self.DP_CORRIDOR_MARGIN = 0.10
        self.DP_W_CENTER = 2.0
        self.DP_W_SMOOTH = 18.0
        self._dp_dbg = {}

        self.LANE_MARGIN = 0.20
        # ✅ 锥桶膨胀/安全距离（横向）：保持适中，避免横向过度膨胀
        # CONE_MARGIN：锥桶边界向外推开的基础距离（横向）
        # CONE_EXTRA_CLEAR：额外安全余量（结合车宽一起计算，横向）
        self.CONE_MARGIN = 0.30
        self.CONE_EXTRA_CLEAR = 0.20

        # ✅ 锥桶前后“纵向膨胀”距离：只在第一个/最后一个锥桶附近扩展
        # 目的：让走廊在锥桶“首尾”前后更早/更晚收敛，而不是横向放大
        self.CONE_S_EXT_BACK = 3.0
        self.CONE_S_EXT_FRONT = 6.0

        # 动态预测障碍物开关
        self.ENABLE_DYNAMIC_BLOCK = True

        self._u_prev = None
        self.WHEELBASE = 3.5 #2.7
        self._max_steer_rad: Optional[float] = None
        self._steer_prev = 0.0

    def _ensure_vehicle_params(self, ego: carla.Vehicle):
        if self._max_steer_rad is not None:
            return
        try:
            pc = ego.get_physics_control()
            max_deg = max(w.max_steer_angle for w in pc.wheels)
            self._max_steer_rad = float(math.radians(max_deg))
        except Exception:
            self._max_steer_rad = float(math.radians(30.0))

    def rebuild_ref_from_ego(self, ego: carla.Actor, step: float = 1.0, max_len: float = 220.0) -> bool:
        ego_wp = self.amap.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            self.ref = None
            return False
        self.ref = LaneRef(self.amap, ego_wp, step=step, max_len=max_len)
        return len(self.ref.P) >= 2

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [list(intervals[0])]
        for a, b in intervals[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return [(float(a), float(b)) for a, b in merged]

    def _speed_linear_lat_extra(self, world: carla.World, actor: carla.Actor) -> float:
        """
        ✅ 基于速度的线性膨胀（不使用固定 speed_k 常量）
        通过 CARLA API 动态获取：速度、车道宽、限速
        extra = lane_width * ratio * scale
        ratio = v / speed_limit_mps （0~1 线性）
        scale 是基于车道宽的缩放系数，避免膨胀过大
        - v 来自 actor.get_velocity()
        - lane_width 来自 waypoint
        - speed_limit 来自 actor.get_speed_limit() 或 waypoint.get_speed_limit()
        """
        try:
            vel = actor.get_velocity()
            v = float(math.hypot(vel.x, vel.y))  # m/s
        except Exception:
            return 0.0

        amap = world.get_map()
        lane_width = 3.5
        speed_limit_kmh = None

        try:
            wp = amap.get_waypoint(actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)
            if wp is not None:
                lane_width = float(getattr(wp, "lane_width", 3.5)) or 3.5
                # waypoint.get_speed_limit() 返回 km/h
                if hasattr(wp, "get_speed_limit"):
                    speed_limit_kmh = float(wp.get_speed_limit())
        except Exception:
            pass

        # actor.get_speed_limit() 也返回 km/h（优先使用车辆自身）
        try:
            if hasattr(actor, "get_speed_limit"):
                speed_limit_kmh = float(actor.get_speed_limit())
        except Exception:
            pass

        # 限速兜底：避免除 0
        if speed_limit_kmh is None or speed_limit_kmh <= 1e-3:
            speed_limit_kmh = 50.0

        speed_limit_mps = float(speed_limit_kmh) / 3.6
        # ✅ 线性比例（0~1），避免速度很高时膨胀无限增大
        ratio = min(1.0, max(0.0, v / max(5.0, speed_limit_mps)))
        # ✅ 以车道宽为尺度，轻度膨胀（经验值，可调）
        scale = 0.35
        extra = float(lane_width) * scale * ratio
        # ✅ 再做一次硬上限，避免异常值
        return float(np.clip(extra, 0.0, 1.2))

    def _predict_blocked_intervals(
        self,
        world: carla.World,
        ego: carla.Actor,
        s_nodes: np.ndarray,
        *,
        T_pred: float = 2.5,
        dt_pred: float = 0.4,
        s_band: float = 3.0,
        veh_lat_buffer: float = 0.6,
        ped_lat_buffer: float = 0.9,
        static_v_thresh: float = 0.2,
    ) -> List[List[Tuple[float, float]]]:
        if self.ref is None:
            return [[] for _ in range(len(s_nodes))]

        blocked: List[List[Tuple[float, float]]] = [[] for _ in range(len(s_nodes))]
        s_nodes = np.asarray(s_nodes, dtype=float)

        # ✅ 只考虑“自车前方20m、同车道”的交通参与者
        ego_loc = ego.get_location()
        ego_wp = world.get_map().get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_road_id = ego_wp.road_id if ego_wp is not None else None
        ego_lane_id = ego_wp.lane_id if ego_wp is not None else None
        s0, _ = self.ref.xy2se(ego_loc.x, ego_loc.y, max_proj_dist=None)
        front_consider_dist = 20.0

        def add_block_at_s(s_val, ey_val, lat_radius):
            s_lo = float(s_val) - float(s_band)
            s_hi = float(s_val) + float(s_band)
            idx = np.where((s_nodes >= s_lo) & (s_nodes <= s_hi))[0]
            if len(idx) == 0:
                return
            lo = float(ey_val) - float(lat_radius)
            up = float(ey_val) + float(lat_radius)
            for ii in idx:
                blocked[ii].append((lo, up))

        for a in world.get_actors():
            if getattr(a, "id", None) == ego.id:
                continue
            t = (getattr(a, "type_id", "") or "").lower()
            is_vehicle = t.startswith("vehicle.")
            is_ped = t.startswith("walker.pedestrian")
            if (not is_vehicle) and (not is_ped):
                continue

            loc0 = a.get_location()

            # ✅ 同车道 + 自车前方20m内的过滤
            if ego_wp is None or s0 is None:
                # 如果无法获取自车车道/投影，直接跳过（避免误判）
                continue
            wp_a = world.get_map().get_waypoint(loc0, project_to_road=True, lane_type=carla.LaneType.Driving)
            if (wp_a is None) or (wp_a.road_id != ego_road_id) or (wp_a.lane_id != ego_lane_id):
                continue
            s_actor, _ = self.ref.xy2se(loc0.x, loc0.y, max_proj_dist=None)
            if s_actor is None or (not (0.0 < (float(s_actor) - float(s0)) <= front_consider_dist)):
                continue

            vel = a.get_velocity()
            vx, vy = float(vel.x), float(vel.y)
            v = float(math.hypot(vx, vy))

            bb = getattr(a, "bounding_box", None)
            half_w = float(getattr(getattr(bb, "extent", None), "y", 0.4))

            # ✅ 动态线性膨胀：速度越快，横向安全半径越大
            # 这里不使用固定 speed_k，而是用 CARLA API 动态计算
            speed_extra = self._speed_linear_lat_extra(world, a)
            lat_radius = half_w + (veh_lat_buffer if is_vehicle else ped_lat_buffer) + speed_extra

            if v < static_v_thresh:
                tf = a.get_transform()
                fwd = tf.get_forward_vector()
                vx, vy = float(fwd.x) * 0.5, float(fwd.y) * 0.5

            for tt in np.arange(0.0, float(T_pred) + 1e-6, float(dt_pred)):
                px = float(loc0.x + vx * tt)
                py = float(loc0.y + vy * tt)
                s_p, ey_p = self.ref.xy2se(px, py, max_proj_dist=None)
                if s_p is None:
                    continue
                add_block_at_s(s_p, ey_p, lat_radius)

        for i in range(len(blocked)):
            blocked[i] = self._merge_intervals(blocked[i])
        return blocked

    def update_corridor(self, world: carla.World, ego: carla.Actor, s_ahead: float = 35.0, ds: float = 1.0, debug_draw: bool = True):
        if ego is None:
            self.corridor = None
            return

        ok = self.rebuild_ref_from_ego(ego, step=1.0, max_len=240.0)
        if not ok:
            self.corridor = None
            self._dp_dbg = {"dp_ok": False, "reason": "ref_build_failed"}
            return

        ego_width = float(ego.bounding_box.extent.y * 2.0)
        required_width = ego_width + 0.40

        out = build_corridor_by_cones_one_side_only(
            world=world, ego=ego, ref=self.ref,
            s_ahead=s_ahead, ds=ds,
            lane_margin=self.LANE_MARGIN,
            cone_margin=self.CONE_MARGIN,
            min_width=1.8,
            cone_extra_clearance=self.CONE_EXTRA_CLEAR,
            cone_s_ext_back=self.CONE_S_EXT_BACK,
            cone_s_ext_front=self.CONE_S_EXT_FRONT,
            expand_adjacent=False,
            required_width=required_width,
        )

        if out is not None:
            min_w = float(out.get("min_width_safe", 999.0))
            if min_w < required_width:
                # ✅ 只在“距离自车近”的位置变窄时，才扩展到相邻车道
                # 避免远处锥桶导致一开始就变成双车道走廊
                try:
                    s_nodes = out["s_nodes"]
                    widths = out["bound_upper_safe"] - out["bound_lower_safe"]
                    min_idx = int(np.argmin(widths))
                    s_min_w = float(s_nodes[min_idx])
                    s0, _ = self.ref.xy2se(
                        ego.get_location().x,
                        ego.get_location().y,
                        max_proj_dist=None
                    )
                    # 触发扩展的“近距离阈值”
                    dist_threshold = 5.0
                    close_enough = (s0 is not None) and ((s_min_w - float(s0)) <= dist_threshold)
                except Exception:
                    close_enough = False

                if close_enough:
                    print(f"[Corridor] single-lane too narrow near ego: min_w={min_w:.2f} < req={required_width:.2f}, expand to adjacent")
                    out2 = build_corridor_by_cones_one_side_only(
                        world=world, ego=ego, ref=self.ref,
                        s_ahead=s_ahead, ds=ds,
                        lane_margin=self.LANE_MARGIN,
                        cone_margin=self.CONE_MARGIN,
                        min_width=1.8,
                        cone_extra_clearance=self.CONE_EXTRA_CLEAR,
                        cone_s_ext_back=self.CONE_S_EXT_BACK,
                        cone_s_ext_front=self.CONE_S_EXT_FRONT,
                        expand_adjacent=True,
                        required_width=required_width,
                    )
                    if out2 is not None:
                        out = out2

        if out is None:
            self.corridor = None
            self._dp_dbg = {"dp_ok": False, "reason": "corridor_build_failed"}
            return

        s_nodes = out["s_nodes"]
        bound_upper_safe = out["bound_upper_safe"]
        bound_lower_safe = out["bound_lower_safe"]

        # ✅ 动态障碍物预测 tube -> blocked_intervals
        blocked = None
        if self.ENABLE_DYNAMIC_BLOCK:
            blocked = self._predict_blocked_intervals(world, ego, s_nodes)

        if self.DP_ENABLE:
            ey_ref, dp_dbg = dp_plan_centerline(
                world=world, ego=ego, ref=self.ref,
                s_nodes=s_nodes,
                bound_upper=bound_upper_safe,
                bound_lower=bound_lower_safe,
                ey_range=self.DP_EY_RANGE,
                dey=self.DP_DEY,
                corridor_margin=self.DP_CORRIDOR_MARGIN,
                W_CENTER=self.DP_W_CENTER,
                W_SMOOTH=self.DP_W_SMOOTH,
                fallback_to_mid=True,
                blocked_intervals=blocked,
            )
            center_ey = ey_ref
            self._dp_dbg = dp_dbg
        else:
            center_ey = 0.5 * (bound_upper_safe + bound_lower_safe)
            self._dp_dbg = {"dp_ok": False, "reason": "DP disabled"}

        self.corridor = SimpleNamespace(
            s=s_nodes,
            left_phys=out["left_phys"],
            right_phys=out["right_phys"],
            bound_upper=out["bound_upper"],
            bound_lower=out["bound_lower"],
            bound_upper_safe=bound_upper_safe,
            bound_lower_safe=bound_lower_safe,
            upper_pts_world=out["upper_pts_world"],
            lower_pts_world=out["lower_pts_world"],
            center_path_ey=center_ey,
        )

        if debug_draw and self.corridor:
            now = world.get_snapshot().timestamp.elapsed_seconds
            if now - self._last_draw_t < self.DRAW_PERIOD:
                return
            self._last_draw_t = now

            dbg = world.debug
            z0 = ego.get_location().z + 0.35

            if self.DRAW_REF_LINE:
                step_idx = max(1, int(3.0 / max(1e-6, self.ref.step)))
                for p0, p1 in zip(self.ref.P[:-1:step_idx], self.ref.P[1::step_idx]):
                    dbg.draw_line(carla.Location(p0[0], p0[1], z0),
                                  carla.Location(p1[0], p1[1], z0),
                                  thickness=0.06, color=COL_REF, life_time=self.DEBUG_LIFE_TIME)

            if self.DRAW_CORRIDOR_EDGES:
                up_pts = self.corridor.upper_pts_world
                lo_pts = self.corridor.lower_pts_world
                for i in range(0, len(up_pts) - 1):
                    # 注释掉紫色左边界线（COL_LEFT）
                    # dbg.draw_line(carla.Location(up_pts[i].x, up_pts[i].y, z0),
                    #               carla.Location(up_pts[i + 1].x, up_pts[i + 1].y, z0),
                    #               thickness=0.12, color=COL_LEFT, life_time=self.DEBUG_LIFE_TIME)
                    # 注释掉绿色右边界线（COL_RIGHT）
                    # dbg.draw_line(carla.Location(lo_pts[i].x, lo_pts[i].y, z0),
                    #               carla.Location(lo_pts[i + 1].x, lo_pts[i + 1].y, z0),
                    #               thickness=0.12, color=COL_RIGHT, life_time=self.DEBUG_LIFE_TIME)
                    pass

            if self.DRAW_DP_CENTER:
                dp_pts = []
                for s_val, ey_c in zip(self.corridor.s, self.corridor.center_path_ey):
                    cx, cy = self.ref.se2xy(s_val, float(ey_c))
                    dp_pts.append(carla.Location(cx, cy, z0))
                for p0, p1 in zip(dp_pts[:-1], dp_pts[1:]):
                    dbg.draw_line(p0, p1, thickness=0.10, color=COL_DP, life_time=self.DEBUG_LIFE_TIME)

    def vehicle_model_frenet(self, x, u):
        vx, ey, yaw_err, s = x
        accel, delta = u
        s_dot = vx * math.cos(yaw_err)
        k_ref = float(self.ref.kappa_at_s(float(s))) if self.ref is not None else 0.0
        ey_dot = vx * math.sin(yaw_err)
        yaw_err_dot = vx * math.tan(delta) / max(1e-6, self.WHEELBASE) - k_ref * s_dot
        vx_dot = accel
        return np.array([vx_dot, ey_dot, yaw_err_dot, s_dot], dtype=float)

    def compute_control(self, ego: carla.Actor, dt: float = 0.05) -> Tuple[float, float, float, Dict[str, Any]]:
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = float(math.hypot(vel.x, vel.y))

        if self.corridor is None or self.ref is None or len(self.corridor.s) < 3:
            return 0.0, 0.0, 0.4, {"opt_ok": False, "reason": "no_corridor"}

        self._ensure_vehicle_params(ego)
        delta_max = float(self._max_steer_rad)
        delta_min = -delta_max

        ego_wp = self.amap.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_w = float(getattr(ego_wp, "lane_width", 3.5)) if ego_wp is not None else 3.5
        max_proj_dist = 1.2 * lane_w + 1.0

        s_now, ey_now = self.ref.xy2se(tf.location.x, tf.location.y, max_proj_dist=max_proj_dist)
        if s_now is None:
            # ✅ 不要 break 主循环：给一个可恢复的 fail，让外层继续 tick
            return 0.0, 0.0, 0.6, {"opt_ok": False, "reason": "ego_out_of_ref", "opt_msg": "projection_fail"}

        ego_yaw_rad = math.radians(tf.rotation.yaw)
        s_idx = int(np.searchsorted(self.ref.s, s_now))
        s_idx = min(max(s_idx, 0), len(self.ref.tang) - 1)
        ref_yaw_rad = math.atan2(self.ref.tang[s_idx, 1], self.ref.tang[s_idx, 0])
        yaw_err_now = ego_yaw_rad - ref_yaw_rad
        yaw_err_now = (yaw_err_now + np.pi) % (2 * np.pi) - np.pi

        x0 = np.array([speed, ey_now, yaw_err_now, s_now], dtype=float)

        # ✅ 增大预测时域：提升大曲率/大角度场景的可跟踪性
        H = 20
        accel_min, accel_max = -5.0, 3.0

        # ✅ 跟踪权重：适度提高横向/航向误差权重
        W_EY_TRACK = 10.0
        W_YAW_TRACK = 12.0
        W_SPEED = 0.5
        W_U = 0.10
        W_DU = 0.20
        W_BOUND = 1500.0
        # ✅ 放松转向变化惩罚，减少“转向响应过慢”
        W_STEER_RATE = 4.0
        W_LAT = 0.06

        def get_bounds_at_s(s):
            up = float(np.interp(s, self.corridor.s, self.corridor.bound_upper_safe))
            lo = float(np.interp(s, self.corridor.s, self.corridor.bound_lower_safe))
            if up < lo:
                mid = 0.5 * (up + lo)
                up = mid + 0.05
                lo = mid - 0.05
            return up, lo

        def ey_target_at_s(s):
            return float(np.interp(s, self.corridor.s, self.corridor.center_path_ey))

        up_now, lo_now = get_bounds_at_s(s_now)
        width_now = float(up_now - lo_now)

        v_ref = float(self.v_ref_base)
        if width_now < 2.6:
            v_ref = min(v_ref, 5.0)
        elif width_now < 3.2:
            v_ref = min(v_ref, 7.0)
        elif width_now < 4.0:
            v_ref = min(v_ref, 9.0)
        # ✅ 基于曲率的速度约束：大弯时主动降速，提升跟踪稳定性
        try:
            kappa_now = abs(float(self.ref.kappa_at_s(float(s_now))))
            if kappa_now > 1e-4:
                # 横向加速度限幅：a_lat = v^2 * kappa
                v_ref = min(v_ref, math.sqrt(2.5 / kappa_now))
        except Exception:
            pass

        # ✅ 仅考虑“同车道、前方20m内”的前车，加入纵向安全距离
        front_gap = None
        front_speed = None
        if ego_wp is not None:
            try:
                world = ego.get_world()
                for a in world.get_actors():
                    if getattr(a, "id", None) == ego.id:
                        continue
                    tname = (getattr(a, "type_id", "") or "").lower()
                    if not tname.startswith("vehicle."):
                        continue

                    loc_a = a.get_location()
                    wp_a = self.amap.get_waypoint(loc_a, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if (wp_a is None) or (wp_a.road_id != ego_wp.road_id) or (wp_a.lane_id != ego_wp.lane_id):
                        continue

                    s_a, _ = self.ref.xy2se(loc_a.x, loc_a.y, max_proj_dist=max_proj_dist)
                    if s_a is None:
                        continue
                    ds = float(s_a) - float(s_now)
                    if not (0.0 < ds <= 20.0):
                        continue

                    if (front_gap is None) or (ds < front_gap):
                        front_gap = ds
                        va = a.get_velocity()
                        front_speed = float(math.hypot(va.x, va.y))
            except Exception:
                pass

        # ✅ 前车距离限速（纵向）：避免追尾
        if front_gap is not None:
            # 反应时间 + 制动距离 + 缓冲（经验值，可调）
            t_react = 1.0
            a_brake = 3.5
            buffer = 2.0
            d_safe = buffer + speed * t_react + (speed * speed) / max(1e-6, 2.0 * a_brake)

            # 用可停车距离限制速度
            d_gap = max(0.0, float(front_gap) - buffer)
            v_stop = math.sqrt(max(0.0, 2.0 * a_brake * d_gap))
            v_ref = min(v_ref, v_stop)

            # 不要明显超过前车速度
            if front_speed is not None:
                v_ref = min(v_ref, front_speed + max(0.0, d_gap) * 0.2)

            # 如果已经小于安全距离，进一步压低速度
            if front_gap < d_safe:
                v_ref = min(v_ref, v_stop)

        if self._u_prev is None or np.shape(self._u_prev) != (H, 2):
            u0 = np.zeros((H, 2), dtype=float)
        else:
            u0 = np.vstack([self._u_prev[1:], self._u_prev[-1:]]).copy()

        u0_flat = u0.reshape(-1)

        def relu(z):
            return z if z > 0.0 else 0.0

        SUB = 3
        h = float(dt) / float(SUB)

        def objective(u_flat):
            u = u_flat.reshape((H, 2))
            cost_u = W_U * (np.sum(u[:, 0] ** 2) + np.sum(u[:, 1] ** 2))
            cost_du = W_DU * (np.sum(np.diff(u[:, 0]) ** 2) + np.sum(np.diff(u[:, 1]) ** 2))

            x = x0.copy()
            cost_track = 0.0
            cost_bound = 0.0
            cost_steer_rate = 0.0
            cost_lat = 0.0

            prev_delta = float(u0[0, 1])

            for k in range(H):
                accel = float(u[k, 0])
                delta = float(u[k, 1])

                ddelta = delta - prev_delta
                cost_steer_rate += W_STEER_RATE * (ddelta ** 2)
                prev_delta = delta

                for _ in range(SUB):
                    x = x + self.vehicle_model_frenet(x, [accel, delta]) * h

                vx_k, ey_k, yaw_k, s_k = float(x[0]), float(x[1]), float(x[2]), float(x[3])

                ey_t = ey_target_at_s(s_k)
                cost_track += W_EY_TRACK * ((ey_k - ey_t) ** 2)
                cost_track += W_YAW_TRACK * (yaw_k ** 2)
                cost_track += W_SPEED * ((vx_k - v_ref) ** 2)

                up, lo = get_bounds_at_s(s_k)
                v_up = relu(ey_k - up)
                v_lo = relu(lo - ey_k)
                cost_bound += W_BOUND * (v_up * v_up + v_lo * v_lo)

                kappa = math.tan(delta) / max(1e-6, self.WHEELBASE)
                a_lat = (vx_k * vx_k) * kappa
                cost_lat += W_LAT * (a_lat * a_lat)

            return cost_u + cost_du + cost_track + cost_bound + cost_steer_rate + cost_lat

        bounds = [(accel_min, accel_max), (delta_min, delta_max)] * H

        res = minimize(
            objective,
            u0_flat,
            bounds=bounds,
            method="SLSQP",
            options={"maxiter": 70, "ftol": 1e-3, "disp": False}
        )

        if (not getattr(res, "success", False)) or (res.x is None) or (not np.all(np.isfinite(res.x))):
            dbg = {
                "s": float(s_now), "ey": float(ey_now),
                "v": float(speed), "v_ref": float(v_ref),
                "lo": float(lo_now), "up": float(up_now),
                "width": float(width_now),
                "dp_ok": bool(self._dp_dbg.get("dp_ok", False)),
                "opt_ok": False,
                "reason": "opt_fail",
                "opt_msg": str(getattr(res, "message", "fail")),
            }
            return 0.0, 0.0, 0.4, dbg

        u_opt = res.x.reshape((H, 2))
        self._u_prev = u_opt.copy()

        accel0 = float(u_opt[0, 0])
        delta0 = float(u_opt[0, 1])

        if accel0 >= 0:
            throttle = float(np.clip(accel0 / accel_max, 0, 1))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip((-accel0) / (-accel_min), 0, 1))

        raw_steer = float(np.clip(delta0 / delta_max, -1, 1))

        # ✅ 提升单步转向变化上限，增强快速转向能力
        max_step = 0.25
        raw_steer = float(np.clip(raw_steer, self._steer_prev - max_step, self._steer_prev + max_step))
        # ✅ 降低转向滤波系数，减少滞后
        alpha = 0.35
        steer = float(alpha * raw_steer + (1.0 - alpha) * self._steer_prev)
        self._steer_prev = steer

        dbg = {
            "s": float(s_now), "ey": float(ey_now),
            "lo": float(lo_now), "up": float(up_now),
            "width": float(width_now),
            "v": float(speed), "v_ref": float(v_ref),
            "delta": float(delta0), "steer": float(steer),
            "throttle": float(throttle), "brake": float(brake),
            "dp_ok": bool(self._dp_dbg.get("dp_ok", False)),
            "opt_ok": True,
            "opt_msg": "ok"
        }
        return throttle, steer, brake, dbg


# ============================================================
# 8) main：跑起来 - 支持多场景选择
# ============================================================
def main(scenario_type: str = "cones"):
    print(f"\n{'='*60}")
    print("  Rule-Based Planner 场景测试")
    print(f"  场景类型: {scenario_type}")
    print(f"{'='*60}\n")

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    amap = world.get_map()

    # TrafficFlowSpawner 保留
    traffic_flow_spawner = TrafficFlowSpawner(client, world, 8000)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    env = None
    logger = None
    scenario = None
    ego = None

    try:
        if scenario_type == "cones":
            print("[场景] 初始化 ConesScenario（锥桶 + 交通流）")
            config = SimpleNamespace(
                cone_num=8,
                cone_step_behind=3.0,
                cone_step_lateral=0.4,
                cone_z_offset=0.5,
                cone_lane_margin=0.25,
                cone_min_gap_from_junction=15.0,
                cone_grid=5.0,
                spawn_min_gap_from_cone=25.0,
                tm_port=8000,
                enable_traffic_flow=True,
            )
            scenario = ConesScenario(world, amap, config)
            if not scenario.setup():
                raise RuntimeError("Cones 场景初始化失败")
            ego, ego_wp = spawn_ego_from_scenario(world, scenario, env=None)

        elif scenario_type == "jaywalker":
            print("[场景] 初始化鬼探头场景（行人突然横穿）")
            config = SimpleNamespace(
                jaywalker_distance=25.0,
                jaywalker_speed=2.5,
                jaywalker_trigger_distance=18.0,
                jaywalker_start_side="random",
                use_occlusion_vehicle=False,
                tm_port=8000,
                enable_traffic_flow=True,
            )
            scenario = JaywalkerScenario(world, amap, config)
            if not scenario.setup():
                raise RuntimeError("鬼探头场景初始化失败")
            ego, ego_wp = spawn_ego_from_scenario(world, scenario, env=None)

        elif scenario_type == "trimma":
            print("[场景] 初始化 Trimma 场景（左右夹击 + 前车）")
            config = SimpleNamespace(
                front_vehicle_distance=18.0,
                side_vehicle_offset=3.0,
                min_lane_count=3,
                tm_port=8000,
                tm_global_distance=2.5,
                front_speed_diff_pct=70.0,
                side_speed_diff_pct=80.0,
                disable_lane_change=True,
                enable_traffic_flow=True,
            )
            scenario = TrimmaScenario(world, amap, config)
            if not scenario.setup():
                raise RuntimeError("Trimma场景初始化失败")
            ego, ego_wp = spawn_ego_from_scenario(world, scenario, env=None)

        elif scenario_type == "construction":
            print("[场景] 初始化施工封道 + 高密度交通流变道场景")
            config = SimpleNamespace(
                construction_distance=30.0,
                construction_length=20.0,
                traffic_density=3.0,
                traffic_speed=8.0,
                min_gap_for_lane_change=12.0,
                construction_type="construction1",
                flow_range=80.0,
                tm_port=8000,
                enable_traffic_flow=True,
            )
            scenario = ConstructionLaneChangeScenario(world, amap, config)
            if not scenario.setup():
                raise RuntimeError("施工场景初始化失败")
            ego, ego_wp = spawn_ego_from_scenario(world, scenario, env=None)

        else:
            raise ValueError(f"未知场景类型: {scenario_type}")

        planner = RuleBasedPlanner(amap, v_ref_base=12.0)
        logger = TelemetryLogger(out_dir=f"logs_rule_based_{scenario_type}")

        dt = 0.05
        frame = 0

        print("\n[开始] 场景运行中... (按 Ctrl+C 停止)\n")

        while True:
            spectator_follow_ego(world, ego)

            planner.update_corridor(world, ego, s_ahead=35.0, ds=1.0, debug_draw=True)
            throttle, steer, brake, dbg = planner.compute_control(ego, dt=dt)

            ctrl = carla.VehicleControl()
            ctrl.throttle = float(throttle)
            ctrl.steer = float(steer)
            ctrl.brake = float(brake)
            ego.apply_control(ctrl)

            world.tick()

            if scenario_type == "jaywalker" and scenario is not None:
                ego_loc = ego.get_location()
                scenario.check_and_trigger(ego_loc)
                scenario.tick_update()

            obs = None
            if env is not None:
                # ⚠️ 注意：如果你的 env.step 内部也 tick，那会双 tick。
                # 你 cones 场景之前能跑，说明你 env.step 很可能只是取观测/更新缓存。
                obs, _ = env.step()

            if dbg and frame % 10 == 0:
                if dbg.get("opt_ok", False):
                    print(f"[CTRL] s={dbg['s']:.1f} ey={dbg['ey']:.2f} | lo={dbg['lo']:.2f} up={dbg['up']:.2f} "
                          f"w={dbg['width']:.2f} | v={dbg['v']:.2f}->{dbg['v_ref']:.2f} "
                          f"| delta={dbg.get('delta', 0.0):.3f} steer={dbg.get('steer', 0.0):.2f} "
                          f"| dp_ok={dbg.get('dp_ok', False)} opt_ok=True | msg={dbg.get('opt_msg','')}")
                else:
                    print(f"[CTRL-FAIL] reason={dbg.get('reason','')} msg={dbg.get('opt_msg','')}")

            logger.log(frame, obs, dbg, planner.ref if planner.ref is not None else None)
            frame += 1

    except KeyboardInterrupt:
        print("\n[Stop] 手动退出。")

    finally:
        print("\n[清理] 正在清理场景...")
        try:
            if logger is not None:
                logger.save_csv()
                logger.plot()
        except Exception as e:
            print(f"[清理] Logger 保存失败: {e}")

        try:
            if scenario is not None:
                scenario.cleanup()
                print("[清理] 场景清理完成")
        except Exception as e:
            print(f"[清理] 场景清理失败: {e}")

        try:
            if ego is not None:
                ego.destroy()
                print("[清理] Ego 车辆销毁完成")
        except Exception as e:
            print(f"[清理] Ego 销毁失败: {e}")

        try:
            if env is not None:
                env.close()
        except Exception as e:
            print(f"[清理] Env 关闭失败: {e}")

        try:
            settings = world.get_settings()
            settings.synchronous_mode = False
            world.apply_settings(settings)
            print("[清理] 已恢复异步模式")
        except Exception as e:
            print(f"[清理] 恢复异步模式失败: {e}")

        print("[清理] 清理完成\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Rule-Based Planner 多场景测试")
    parser.add_argument(
        "--scenario",
        type=str,
        default="cones",
        choices=["cones", "jaywalker", "trimma", "construction"],
        help="场景类型: cones(锥桶), jaywalker(鬼探头), trimma(左右夹击), construction(施工变道)"
    )
    args = parser.parse_args()
    main(scenario_type=args.scenario)
