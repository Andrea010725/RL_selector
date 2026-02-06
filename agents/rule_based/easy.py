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

# 交通流工具
sys.path.append("/home/ajifang/Driveadapter_2/tools")
from custom_eval import TrafficFlowSpawner

from vis_debug import TelemetryLogger


# ============================================================
# 颜色工具
# ============================================================
def _col(r, g, b):
    return carla.Color(int(r), int(g), int(b))


COL_REF = _col(200, 200, 200)  # 灰：参考线
COL_LEFT = _col(255, 0, 255)  # 紫：物理左边界
COL_RIGHT = _col(50, 220, 50)  # 绿：物理右边界
COL_DP = _col(255, 255, 0)  # 黄：DP/NMPC跟踪中心线


# ============================================================
# 1) LaneRef：局部参考线 + 方向一致性校验 (修复横向生成问题)
# ============================================================
class LaneRef:
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len: float = 200.0):
        pts, wps = [], []
        wp = seed_wp

        dist = 0.0
        pts.append((wp.transform.location.x, wp.transform.location.y))
        wps.append(wp)

        # 用于防止回头的阈值 (cos(theta) > 0.5 表示夹角小于 60 度)
        COS_THRESH = 0.5

        while dist < max_len:
            nxts = wp.next(step)
            if not nxts:
                break

            # ------------------------------------------------------
            # ✅ [核心修复] 在多个后续点中，寻找方向最一致的那个
            # 解决 Carla 在路口/连接处 next() 返回横向车道导致黄线横置的问题
            # ------------------------------------------------------
            best_wp = None
            max_dot = -1.0

            # 获取当前点的切线方向
            fwd = wp.transform.get_forward_vector()

            for n_cand in nxts:
                # 必须是车道类型（过滤掉人行道/路肩等）
                if n_cand.lane_type != carla.LaneType.Driving:
                    continue

                # 计算位移向量
                vec = n_cand.transform.location - wp.transform.location
                norm = math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)
                if norm < 1e-3: continue

                # 计算点积 (Cosine Similarity)
                dot = (vec.x * fwd.x + vec.y * fwd.y + vec.z * fwd.z) / norm

                # 找到最顺路的方向
                if dot > max_dot:
                    max_dot = dot
                    best_wp = n_cand

            # 如果最好的方向都拐得太急（比如横向道路，dot 接近 0），则停止生成
            if best_wp is None or max_dot < COS_THRESH:
                break

            wp = best_wp
            # ------------------------------------------------------

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

        v = xy - P[:-1]  # [N-1,2]
        seg = P[1:] - P[:-1]  # [N-1,2]
        seg_len2 = (seg[:, 0] ** 2 + seg[:, 1] ** 2) + 1e-9

        t = np.clip((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:, None]
        dist2 = np.sum((proj - xy[None, :]) ** 2, axis=1)

        i = int(np.argmin(dist2))
        min_dist = float(math.sqrt(dist2[i]))
        return i, float(t[i]), proj[i], min_dist

    def xy2se(self, x: float, y: float, max_proj_dist: Optional[float] = None) -> Tuple[
        Optional[float], Optional[float]]:
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
# 2) spawn ego
# ============================================================
def spawn_ego_from_scenario(world: carla.World, scenario, env: Optional[HighwayEnv] = None) -> Tuple[
    carla.Actor, carla.Waypoint]:
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
# 小工具：从 wp_center 获取 lane 边界点
# ============================================================
def _lane_bound_points(wp: carla.Waypoint):
    c = wp.transform.location
    right_vec = wp.transform.get_right_vector()
    w = float(getattr(wp, "lane_width", 3.5)) or 3.5
    half = 0.5 * w
    left_edge = carla.Location(x=c.x - right_vec.x * half, y=c.y - right_vec.y * half, z=c.z)
    right_edge = carla.Location(x=c.x + right_vec.x * half, y=c.y + right_vec.y * half, z=c.z)
    return left_edge, right_edge


# ============================================================
# 5) Corridor
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

    ego_wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_road_id = ego_wp.road_id if ego_wp is not None else None
    ego_lane_id = ego_wp.lane_id if ego_wp is not None else None

    allowed_lane_ids = set()
    if ego_wp is not None:
        allowed_lane_ids.add(ego_lane_id)
        if expand_adjacent:
            lwp = ego_wp.get_left_lane()
            rwp = ego_wp.get_right_lane()
            if lwp is not None and lwp.lane_type == carla.LaneType.Driving and (lwp.lane_id * ego_lane_id > 0):
                allowed_lane_ids.add(lwp.lane_id)
            if rwp is not None and rwp.lane_type == carla.LaneType.Driving and (rwp.lane_id * ego_lane_id > 0):
                allowed_lane_ids.add(rwp.lane_id)

    front_consider_dist = 20.0
    s_nodes = np.arange(float(s0), float(s0) + float(s_ahead), float(ds))

    # 1) 外侧边界（单车道或扩到相邻车道）
    left_outer_ey = np.zeros(len(s_nodes), dtype=float)
    right_outer_ey = np.zeros(len(s_nodes), dtype=float)
    left_outer_world = []
    right_outer_world = []

    for i, s in enumerate(s_nodes):
        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
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
        # ✅ [修复] 只要物理投影有效，就算障碍物，不再校验 Road ID
        s_cone, ey_cone = ref.xy2se(loc.x, loc.y, max_proj_dist=None)

        if s_cone is not None and (0 < s_cone - s0 < s_ahead + 20.0):
            # 仅用于排序，取 wp 的 lane_id
            wp = amap.get_waypoint(loc, project_to_road=True)
            if wp:
                cones.append((float(s_cone), float(ey_cone), loc, int(wp.road_id), int(wp.lane_id)))

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

    # 根据“锥桶所在车道”选择通行侧
    cones_sorted = sorted(cones, key=lambda x: abs(x[0] - float(s0)))
    cone_ref = cones_sorted[0]
    cone_road_id = cone_ref[3]
    cone_lane_id = cone_ref[4]

    cone_wp = amap.get_waypoint(cone_ref[2], project_to_road=True, lane_type=carla.LaneType.Driving)
    has_left = False
    has_right = False
    if cone_wp is not None:
        lwp = cone_wp.get_left_lane()
        rwp = cone_wp.get_right_lane()
        if lwp is not None and lwp.lane_type == carla.LaneType.Driving and (lwp.lane_id * cone_lane_id > 0):
            has_left = True
        if rwp is not None and rwp.lane_type == carla.LaneType.Driving and (rwp.lane_id * cone_lane_id > 0):
            has_right = True

    # 3) 连续 cone 边界：按 s 插值
    cones_side = [(sc, ey, loc) for (sc, ey, loc, rid, lid) in cones if (rid == cone_road_id and lid == cone_lane_id)]
    cones_side.sort(key=lambda x: x[0])
    sc_list = np.array([c[0] for c in cones_side], dtype=float)
    ey_list = np.array([c[1] for c in cones_side], dtype=float)

    if has_right:
        corridor_side = "RIGHT"
    elif has_left:
        corridor_side = "LEFT"
    else:
        mean_ey = float(np.mean(ey_list)) if len(ey_list) > 0 else 0.0
        corridor_side = "RIGHT" if mean_ey >= 0.0 else "LEFT"

    mode = corridor_side

    cone_boundary_ey = np.full(len(s_nodes), np.nan, dtype=float)
    if len(sc_list) >= 2:
        s_min, s_max = float(sc_list[0]), float(sc_list[-1])
        mask_core = (s_nodes >= s_min) & (s_nodes <= s_max)
        cone_boundary_ey[mask_core] = np.interp(s_nodes[mask_core], sc_list, ey_list)

        s_min_ext = s_min - max(0.0, float(cone_s_ext_back))
        s_max_ext = s_max + max(0.0, float(cone_s_ext_front))
        mask_ext = (s_nodes >= s_min_ext) & (s_nodes <= s_max_ext)

        if np.any(s_nodes < s_min):
            mask_back = (s_nodes >= s_min_ext) & (s_nodes < s_min)
            cone_boundary_ey[mask_back] = float(ey_list[0])
        if np.any(s_nodes > s_max):
            mask_front = (s_nodes > s_max) & (s_nodes <= s_max_ext)
            cone_boundary_ey[mask_front] = float(ey_list[-1])
        mask = mask_ext
    else:
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

    if np.any(mask):
        smoothed = _smooth_nan_aware(cone_boundary_ey, k=7)
        cone_boundary_ey = np.where(mask, smoothed, np.nan)

    # 4) 把 cone 边界按物理方向推开 cone_margin
    for i, s in enumerate(s_nodes):
        ey = cone_boundary_ey[i]
        if not np.isfinite(ey):
            continue

        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
        if wp_center is None:
            continue

        right_vec = wp_center.transform.get_right_vector()
        px, py = ref.se2xy(float(s), float(ey))

        if corridor_side == "RIGHT":
            p = carla.Location(x=px + right_vec.x * cone_margin, y=py + right_vec.y * cone_margin,
                               z=wp_center.transform.location.z)
        else:
            p = carla.Location(x=px - right_vec.x * cone_margin, y=py - right_vec.y * cone_margin,
                               z=wp_center.transform.location.z)

        _, ey_in = ref.xy2se(p.x, p.y, max_proj_dist=None)
        cone_boundary_ey[i] = float(ey_in)

    # 4.5) 目标车道外侧边界
    target_outer_ey = np.zeros(len(s_nodes), dtype=float)
    target_outer_world = []
    for i, s in enumerate(s_nodes):
        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
        if wp_center is None:
            if corridor_side == "RIGHT":
                target_outer_ey[i] = right_outer_ey[i]
                target_outer_world.append(right_outer_world[i])
            else:
                target_outer_ey[i] = left_outer_ey[i]
                target_outer_world.append(left_outer_world[i])
            continue

        right_vec = wp_center.transform.get_right_vector()
        if corridor_side == "RIGHT":
            wp_t = wp_center.get_right_lane()
            if wp_t is not None and wp_t.lane_type == carla.LaneType.Driving and (wp_t.lane_id * wp_center.lane_id > 0):
                _, edge = _lane_bound_points(wp_t)
            else:
                _, edge = _lane_bound_points(wp_center)
            edge_in = carla.Location(x=edge.x - right_vec.x * lane_margin,
                                     y=edge.y - right_vec.y * lane_margin,
                                     z=edge.z)
        else:
            wp_t = wp_center.get_left_lane()
            if wp_t is not None and wp_t.lane_type == carla.LaneType.Driving and (wp_t.lane_id * wp_center.lane_id > 0):
                edge, _ = _lane_bound_points(wp_t)
            else:
                edge, _ = _lane_bound_points(wp_center)
            edge_in = carla.Location(x=edge.x + right_vec.x * lane_margin,
                                     y=edge.y + right_vec.y * lane_margin,
                                     z=edge.z)

        _, ey_t = ref.xy2se(edge_in.x, edge_in.y, max_proj_dist=None)
        target_outer_ey[i] = float(ey_t)
        target_outer_world.append(edge_in)

    # 5) 生成 corridor 两端
    endA_ey = np.zeros(len(s_nodes), dtype=float)
    endB_ey = np.zeros(len(s_nodes), dtype=float)
    endA_world, endB_world = [], []
    other_side_ey = np.zeros(len(s_nodes), dtype=float)

    for i, s in enumerate(s_nodes):
        cey = cone_boundary_ey[i]
        if not np.isfinite(cey):
            a_ey = left_outer_ey[i]
            b_ey = right_outer_ey[i]
            other_side_ey[i] = (right_outer_ey[i] if corridor_side == "RIGHT" else left_outer_ey[i])
        else:
            if corridor_side == "RIGHT":
                a_ey = cey
                b_ey = target_outer_ey[i]
                other_side_ey[i] = target_outer_ey[i]
            else:
                a_ey = target_outer_ey[i]
                b_ey = cey
                other_side_ey[i] = target_outer_ey[i]

        endA_ey[i] = float(a_ey)
        endB_ey[i] = float(b_ey)
        ax, ay = ref.se2xy(float(s), float(a_ey))
        bx, by = ref.se2xy(float(s), float(b_ey))
        endA_world.append(carla.Location(x=ax, y=ay, z=0.0))
        endB_world.append(carla.Location(x=bx, y=by, z=0.0))

    # 物理左/右
    left_phys = np.zeros(len(s_nodes), dtype=float)
    right_phys = np.zeros(len(s_nodes), dtype=float)
    left_pts_world, right_pts_world = [], []

    for i, s in enumerate(s_nodes):
        cx, cy = ref.se2xy(float(s), 0.0)
        wp_center = amap.get_waypoint(carla.Location(x=cx, y=cy, z=0.0), project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
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
        if not np.isfinite(cey):
            continue
        if corridor_side == "RIGHT":
            bound_upper_safe[i] = min(bound_upper_safe[i], cey - cone_clear)
        else:
            bound_lower_safe[i] = max(bound_lower_safe[i], cey + cone_clear)

        if bound_upper_safe[i] <= bound_lower_safe[i]:
            mid = 0.5 * (bound_upper[i] + bound_lower[i])
            bound_upper_safe[i] = mid + 0.3
            bound_lower_safe[i] = mid - 0.3

    # 车辆挤占（静态当前帧）
    car_clear_lat_base = 0.5 * ego_width + 0.30
    car_influence_s = 8.0
    for a in world.get_actors():
        if getattr(a, "id", None) == ego.id:
            continue
        tname = (getattr(a, "type_id", "") or "").lower()
        if not tname.startswith("vehicle."):
            continue
        loc = a.get_location()

        # ✅ [修复] 只要物理上在参考线投影范围内，都算障碍，不检查 RoadID
        s_car, ey_car = ref.xy2se(loc.x, loc.y, max_proj_dist=None)
        if s_car is None:
            continue

        if not (0.0 < (float(s_car) - float(s0)) <= front_consider_dist):
            continue

        # 横向距离太大（比如对向车道）过滤
        if abs(ey_car) > 6.0:
            continue

        try:
            vel = a.get_velocity()
            v = float(math.hypot(vel.x, vel.y))
        except Exception:
            v = 0.0

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
                bound_upper_safe[i] = min(bound_upper_safe[i], float(ey_car) - car_clear_lat)
            else:
                bound_lower_safe[i] = max(bound_lower_safe[i], float(ey_car) + car_clear_lat)

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
# 6) DP：生成 center_path_ey（黄色线）
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
        W_PREF: float = 6.0,
        preferred_ey: Optional[np.ndarray] = None,  # len==Ns
        fallback_to_mid: bool = True,
        blocked_intervals: Optional[List[List[Tuple[float, float]]]] = None,  # len==Ns, each=[(lo,up)...]
        K_MAX: int = 4,
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

    # preferred_ey fallback
    if preferred_ey is None:
        preferred_ey = 0.5 * (bound_upper + bound_lower)
    preferred_ey = np.asarray(preferred_ey, dtype=float)
    if len(preferred_ey) != Ns:
        preferred_ey = np.resize(preferred_ey, Ns)

    for i in range(Ns):
        lo = float(bound_lower[i]) + corridor_margin
        up = float(bound_upper[i]) - corridor_margin
        if up <= lo:
            lo = float(bound_lower[i])
            up = float(bound_upper[i])

        valid = (ey_grid >= lo) & (ey_grid <= up)

        # ✅ [修复] 强制起点有效：确保 DP 路径一定从车底出发
        if i == 0:
            valid[j_start] = True

        if blocked_intervals is not None and i < len(blocked_intervals):
            for (blo, bup) in blocked_intervals[i]:
                valid &= ~((ey_grid >= float(blo)) & (ey_grid <= float(bup)))

        # 这一步 cost：既考虑 corridor 中心，也考虑 preferred
        mid = 0.5 * (lo + up)
        pref = float(preferred_ey[i])

        if not np.any(valid):
            j_mid = int(np.argmin(np.abs(ey_grid - mid)))
            valid[:] = False
            valid[j_mid] = True

        cost[i, valid] = (
                W_CENTER * ((ey_grid[valid] - mid) ** 2) +
                W_PREF * ((ey_grid[valid] - pref) ** 2)
        )

        # ✅ [修复] 起点代价设为0
        if i == 0:
            cost[i, j_start] = 0.0

    if not np.isfinite(cost[0, j_start]) or cost[0, j_start] >= INF * 0.5:
        if fallback_to_mid:
            return 0.5 * (bound_upper + bound_lower), {"dp_ok": False, "reason": "start_infeasible"}
        return np.zeros(Ns), {"dp_ok": False, "reason": "start_infeasible"}

    dp = np.full((Ns, Ny), INF, dtype=float)
    prev = np.full((Ns, Ny), -1, dtype=int)
    dp[0, j_start] = cost[0, j_start]

    K = int(max(1, K_MAX))
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
# 7) RuleBasedPlanner
# ============================================================
class RuleBasedPlanner:
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
        self.DP_W_SMOOTH = 16.0
        self.DP_W_PREF = 12.0
        self._dp_dbg = {}

        self.LANE_MARGIN = 0.25

        self.CONE_MARGIN = 0.40
        self.CONE_EXTRA_CLEAR = 0.60   #0.25
        self.CONE_S_EXT_BACK = 20.0 #3.0
        self.CONE_S_EXT_FRONT = 8.0  #6.0

        self.ENABLE_DYNAMIC_BLOCK = True

        self._u_prev = None
        self.WHEELBASE = 2.7
        self._max_steer_rad: Optional[float] = None
        self._steer_prev = 0.0

        self._lc_state = "IDLE"  # IDLE / COMMIT
        self._lc_target_lane_id: Optional[int] = None
        self._lc_commit_t = 0.0
        self._lc_min_commit_time = 1.2
        self._lc_finish_err = 0.25

        self.enable_overtake = False
        self._overtake_active = False
        self._front_slow_time = 0.0
        self._overtake_front_id = None
        self._trimma_lock_lane = False
        self._trimma_locked_lane_id = None

        # ✅ [新增] 核心开关：是否允许因为拥堵自动变道
        # cones 设为 False，其他 True
        self.enable_auto_lane_change = True

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

    def _get_lane_wp_by_id(self, wp_center: carla.Waypoint, lane_id: int) -> Optional[carla.Waypoint]:
        if wp_center is None:
            return None
        if int(wp_center.lane_id) == int(lane_id):
            return wp_center

        if wp_center.lane_id * lane_id <= 0:
            return None

        wp = wp_center
        guard = 0
        while wp is not None and guard < 8:
            if int(wp.lane_id) == int(lane_id):
                return wp
            if int(lane_id) > int(wp.lane_id):
                nxt = wp.get_left_lane()
                if nxt is None or nxt.lane_type != carla.LaneType.Driving or (nxt.lane_id * wp.lane_id <= 0):
                    nxt = wp.get_right_lane()
            else:
                nxt = wp.get_right_lane()
                if nxt is None or nxt.lane_type != carla.LaneType.Driving or (nxt.lane_id * wp.lane_id <= 0):
                    nxt = wp.get_left_lane()
            wp = nxt
            guard += 1
        return None

    def _lane_band_at_s(self, s_val: float, lane_id: int) -> Optional[Tuple[float, float, float]]:
        if self.ref is None:
            return None
        cx, cy = self.ref.se2xy(float(s_val), 0.0)
        wp_center = self.amap.get_waypoint(
            carla.Location(x=cx, y=cy, z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp_center is None:
            return None
        wp_lane = self._get_lane_wp_by_id(wp_center, lane_id)
        if wp_lane is None:
            return None

        left_edge, right_edge = _lane_bound_points(wp_lane)
        right_vec = wp_lane.transform.get_right_vector()

        left_in = carla.Location(
            x=left_edge.x + right_vec.x * self.LANE_MARGIN,
            y=left_edge.y + right_vec.y * self.LANE_MARGIN,
            z=left_edge.z
        )
        right_in = carla.Location(
            x=right_edge.x - right_vec.x * self.LANE_MARGIN,
            y=right_edge.y - right_vec.y * self.LANE_MARGIN,
            z=right_edge.z
        )

        _, ey_left = self.ref.xy2se(left_in.x, left_in.y, max_proj_dist=None)
        _, ey_right = self.ref.xy2se(right_in.x, right_in.y, max_proj_dist=None)
        if ey_left is None or ey_right is None:
            return None
        lo = float(min(ey_left, ey_right))
        up = float(max(ey_left, ey_right))

        c = wp_lane.transform.location
        _, ey_c = self.ref.xy2se(c.x, c.y, max_proj_dist=None)
        if ey_c is None:
            ey_c = 0.5 * (lo + up)

        return lo, up, float(ey_c)

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
                if hasattr(wp, "get_speed_limit"):
                    speed_limit_kmh = float(wp.get_speed_limit())
        except Exception:
            pass

        try:
            if hasattr(actor, "get_speed_limit"):
                speed_limit_kmh = float(actor.get_speed_limit())
        except Exception:
            pass

        if speed_limit_kmh is None or speed_limit_kmh <= 1e-3:
            speed_limit_kmh = 50.0

        speed_limit_mps = float(speed_limit_kmh) / 3.6
        ratio = min(1.0, max(0.0, v / max(5.0, speed_limit_mps)))
        scale = 0.35
        extra = float(lane_width) * scale * ratio
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
            include_adjacent: bool = False,
    ) -> List[List[Tuple[float, float]]]:
        if self.ref is None:
            return [[] for _ in range(len(s_nodes))]

        blocked: List[List[Tuple[float, float]]] = [[] for _ in range(len(s_nodes))]
        s_nodes = np.asarray(s_nodes, dtype=float)

        ego_loc = ego.get_location()
        ego_wp = world.get_map().get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_road_id = ego_wp.road_id if ego_wp is not None else None
        ego_lane_id = ego_wp.lane_id if ego_wp is not None else None
        s0, _ = self.ref.xy2se(ego_loc.x, ego_loc.y, max_proj_dist=None)
        front_consider_dist = 20.0

        allowed_lane_ids = set()
        if ego_wp is not None:
            allowed_lane_ids.add(ego_lane_id)
            if include_adjacent:
                lwp = ego_wp.get_left_lane()
                rwp = ego_wp.get_right_lane()
                if lwp is not None and lwp.lane_type == carla.LaneType.Driving and (lwp.lane_id * ego_lane_id > 0):
                    allowed_lane_ids.add(lwp.lane_id)
                if rwp is not None and rwp.lane_type == carla.LaneType.Driving and (rwp.lane_id * ego_lane_id > 0):
                    allowed_lane_ids.add(rwp.lane_id)

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

            # ✅ [修复] 移除 ID 检查，只做几何投影
            s_actor, ey_actor = self.ref.xy2se(loc0.x, loc0.y, max_proj_dist=None)
            if s_actor is None or (not (0.0 < (float(s_actor) - float(s0)) <= front_consider_dist)):
                continue

            # 过滤太远的对向车
            if abs(ey_actor) > 8.0:
                continue

            vel = a.get_velocity()
            vx, vy = float(vel.x), float(vel.y)
            v = float(math.hypot(vx, vy))

            bb = getattr(a, "bounding_box", None)
            half_w = float(getattr(getattr(bb, "extent", None), "y", 0.4))

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

    # ------------------------------------------------------------
    # lane-change 决策
    # ------------------------------------------------------------
    def _update_lane_change_commit(self, world: carla.World, ego: carla.Actor, out: Dict[str, Any],
                                   ego_wp: Optional[carla.Waypoint]):
        now = world.get_snapshot().timestamp.elapsed_seconds
        if ego_wp is None:
            self._lc_state = "IDLE"
            self._lc_target_lane_id = None
            return

        mode = out.get("mode", "NONE")
        expanded = bool(out.get("expanded", False))

        s_nodes = out["s_nodes"]
        bu = out["bound_upper_safe"]
        bl = out["bound_lower_safe"]

        ego_loc = ego.get_location()
        s0, _ = self.ref.xy2se(ego_loc.x, ego_loc.y, max_proj_dist=None) if self.ref is not None else (None, None)
        if s0 is None:
            return

        near_mask = (s_nodes >= float(s0)) & (s_nodes <= float(s0) + 12.0)
        if not np.any(near_mask):
            near_mask = np.ones_like(s_nodes, dtype=bool)
        widths = (bu - bl)
        min_w_near = float(np.min(widths[near_mask]))

        ego_width = float(ego.bounding_box.extent.y * 2.0)
        required_width = ego_width + 0.45

        # ✅ [新增] 只有在允许自动变道时，才因为路窄而 COMMIT
        need_change = expanded or (self.enable_auto_lane_change and (min_w_near < required_width))

        if self._lc_state == "IDLE":
            if need_change:
                target_lane_id = None
                if mode == "RIGHT":
                    rwp = ego_wp.get_right_lane()
                    if rwp is not None and rwp.lane_type == carla.LaneType.Driving and (
                            rwp.lane_id * ego_wp.lane_id > 0):
                        target_lane_id = int(rwp.lane_id)
                elif mode == "LEFT":
                    lwp = ego_wp.get_left_lane()
                    if lwp is not None and lwp.lane_type == carla.LaneType.Driving and (
                            lwp.lane_id * ego_wp.lane_id > 0):
                        target_lane_id = int(lwp.lane_id)

                if target_lane_id is None and need_change:
                    rwp = ego_wp.get_right_lane()
                    lwp = ego_wp.get_left_lane()
                    if rwp is not None and rwp.lane_type == carla.LaneType.Driving and (
                            rwp.lane_id * ego_wp.lane_id > 0):
                        target_lane_id = int(rwp.lane_id)
                    elif lwp is not None and lwp.lane_type == carla.LaneType.Driving and (
                            lwp.lane_id * ego_wp.lane_id > 0):
                        target_lane_id = int(lwp.lane_id)

                if target_lane_id is not None:
                    self._lc_state = "COMMIT"
                    self._lc_target_lane_id = int(target_lane_id)
                    self._lc_commit_t = float(now)

        # COMMIT 结束条件
        if self._lc_state == "COMMIT" and self._lc_target_lane_id is not None:
            try:
                ego_loc = ego.get_location()
                s_now, ey_now = self.ref.xy2se(ego_loc.x, ego_loc.y, max_proj_dist=None)
                if s_now is not None and ey_now is not None:
                    band = self._lane_band_at_s(float(s_now), int(self._lc_target_lane_id))
                    if band is not None:
                        lo, up, ey_c = band
                        err = abs(float(ey_now) - float(ey_c))
                        if (now - self._lc_commit_t) >= self._lc_min_commit_time and err <= self._lc_finish_err:
                            self._lc_state = "IDLE"
                            self._lc_target_lane_id = None
            except Exception:
                pass

    # ------------------------------------------------------------
    # Trimma 超车
    # ------------------------------------------------------------
    def _update_trimma_overtake(self, world: carla.World, ego: carla.Actor):
        if not self.enable_overtake or self.ref is None:
            self._overtake_active = False
            self._front_slow_time = 0.0
            self._overtake_front_id = None
            return

        ego_loc = ego.get_location()
        ego_wp = self.amap.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            self._overtake_active = False
            self._front_slow_time = 0.0
            self._overtake_front_id = None
            return

        s0, _ = self.ref.xy2se(ego_loc.x, ego_loc.y, max_proj_dist=None)
        if s0 is None:
            self._overtake_active = False
            self._front_slow_time = 0.0
            self._overtake_front_id = None
            return

        front_actor = None
        front_gap = None
        for a in world.get_actors():
            if getattr(a, "id", None) == ego.id:
                continue
            tname = (getattr(a, "type_id", "") or "").lower()
            if not tname.startswith("vehicle."):
                continue
            loc = a.get_location()
            wp_a = self.amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if (wp_a is None) or (wp_a.road_id != ego_wp.road_id) or (wp_a.lane_id != ego_wp.lane_id):
                continue
            s_a, _ = self.ref.xy2se(loc.x, loc.y, max_proj_dist=None)
            if s_a is None:
                continue
            ds = float(s_a) - float(s0)
            if ds <= 0.0:
                continue
            if front_gap is None or ds < front_gap:
                front_gap = ds
                front_actor = a

        if front_actor is None:
            self._overtake_active = False
            self._front_slow_time = 0.0
            self._overtake_front_id = None
            return

        vel = front_actor.get_velocity()
        v_front = float(math.hypot(vel.x, vel.y))
        low_speed_thresh = max(2.0, 0.6 * float(self.v_ref_base))
        trigger_range = 18.0

        dt = world.get_snapshot().timestamp.delta_seconds
        if front_gap is not None and front_gap <= trigger_range and v_front < low_speed_thresh:
            self._front_slow_time += dt
        else:
            self._front_slow_time = 0.0

        if self._front_slow_time >= 1.0:
            self._overtake_active = True
            self._overtake_front_id = int(front_actor.id)

        if self._overtake_active:
            if front_gap is None or front_gap > 40.0:
                self._overtake_active = False
                self._overtake_front_id = None
            else:
                self._overtake_front_id = int(front_actor.id)

        if self.enable_overtake and (not self._overtake_active):
            if ego_wp is not None:
                self._trimma_lock_lane = True
                self._trimma_locked_lane_id = int(ego_wp.lane_id)

    # ------------------------------------------------------------
    # update_corridor
    # ------------------------------------------------------------
    def update_corridor(self, world: carla.World, ego: carla.Actor, s_ahead: float = 35.0, ds: float = 1.0,
                        debug_draw: bool = True):
        if ego is None:
            self.corridor = None
            return

        try:
            v_ego = ego.get_velocity()
            ego_speed = float(math.hypot(v_ego.x, v_ego.y))
        except Exception:
            ego_speed = 0.0

        ok = self.rebuild_ref_from_ego(ego, step=1.0, max_len=240.0)
        if not ok:
            self.corridor = None
            self._dp_dbg = {"dp_ok": False, "reason": "ref_build_failed"}
            return

        ego_wp = self.amap.get_waypoint(ego.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving)

        self._update_trimma_overtake(world, ego)

        ego_width = float(ego.bounding_box.extent.y * 2.0)
        required_width = ego_width + 0.45

        expand_adjacent = bool(self._lc_state == "COMMIT" or self._overtake_active)

        out = build_corridor_by_cones_one_side_only(
            world=world, ego=ego, ref=self.ref,
            s_ahead=s_ahead, ds=ds,
            lane_margin=self.LANE_MARGIN,
            cone_margin=self.CONE_MARGIN,
            min_width=1.8,
            cone_extra_clearance=self.CONE_EXTRA_CLEAR,
            cone_s_ext_back=self.CONE_S_EXT_BACK,
            cone_s_ext_front=self.CONE_S_EXT_FRONT,
            expand_adjacent=expand_adjacent,
            required_width=required_width,
        )

        if out is None:
            self.corridor = None
            self._dp_dbg = {"dp_ok": False, "reason": "corridor_build_failed"}
            return

        if not expand_adjacent:
            try:
                widths = out["bound_upper_safe"] - out["bound_lower_safe"]
                min_w = float(np.min(widths))

                # ✅ [新增] 只有开启了 enable_auto_lane_change，才允许因为路窄而扩宽
                if self.enable_auto_lane_change and min_w < required_width:
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
                        expand_adjacent = True
            except Exception:
                pass

        self._update_lane_change_commit(world, ego, out, ego_wp)

        s_nodes = out["s_nodes"]
        bound_upper_safe = out["bound_upper_safe"]
        bound_lower_safe = out["bound_lower_safe"]

        blocked = None
        if self.ENABLE_DYNAMIC_BLOCK:
            blocked = self._predict_blocked_intervals(
                world, ego, s_nodes,
                include_adjacent=bool(expand_adjacent)
            )

        preferred_ey = np.zeros(len(s_nodes), dtype=float)

        desired_lane_id = None
        if ego_wp is not None:
            desired_lane_id = int(ego_wp.lane_id)

        if self.enable_overtake and self._overtake_active and ego_wp is not None:
            rwp = ego_wp.get_right_lane()
            lwp = ego_wp.get_left_lane()
            if rwp is not None and rwp.lane_type == carla.LaneType.Driving and (rwp.lane_id * ego_wp.lane_id > 0):
                desired_lane_id = int(rwp.lane_id)
            elif lwp is not None and lwp.lane_type == carla.LaneType.Driving and (lwp.lane_id * ego_wp.lane_id > 0):
                desired_lane_id = int(lwp.lane_id)

        if self._lc_state == "COMMIT" and self._lc_target_lane_id is not None:
            desired_lane_id = int(self._lc_target_lane_id)

        if desired_lane_id is not None:
            for i, s_val in enumerate(s_nodes):
                band = self._lane_band_at_s(float(s_val), int(desired_lane_id))
                if band is None:
                    preferred_ey[i] = 0.0
                else:
                    _, _, ey_c = band
                    preferred_ey[i] = float(ey_c)
        else:
            preferred_ey[:] = 0.0

        if self._lc_state == "COMMIT" or self._overtake_active:
            dp_W_center = 2.0
            dp_W_smooth = 14.0
            dp_W_pref = 16.0
            dp_Kmax = 4
        else:
            dp_W_center = 2.0
            dp_W_smooth = 18.0
            dp_W_pref = 18.0
            dp_Kmax = 3

        if self.DP_ENABLE:
            center_ey, dp_dbg = dp_plan_centerline(
                world=world, ego=ego, ref=self.ref,
                s_nodes=s_nodes,
                bound_upper=bound_upper_safe,
                bound_lower=bound_lower_safe,
                ey_range=self.DP_EY_RANGE,
                dey=self.DP_DEY,
                corridor_margin=self.DP_CORRIDOR_MARGIN,
                W_CENTER=dp_W_center,
                W_SMOOTH=dp_W_smooth,
                W_PREF=dp_W_pref,
                preferred_ey=preferred_ey,
                fallback_to_mid=True,
                blocked_intervals=blocked,
                K_MAX=dp_Kmax,
            )

            try:
                ker = np.ones(7, dtype=float) / 7.0
                sm = np.convolve(center_ey, ker, mode="same")
                center_ey = sm
            except Exception:
                pass

            for i in range(len(s_nodes)):
                lo = float(bound_lower_safe[i]) + 0.03
                up = float(bound_upper_safe[i]) - 0.03
                if up > lo:
                    center_ey[i] = float(np.clip(center_ey[i], lo, up))

            self._dp_dbg = dp_dbg
        else:
            center_ey = preferred_ey.copy()
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
            desired_lane_id=desired_lane_id,
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

            if self.DRAW_DP_CENTER:
                dp_pts = []
                for s_val, ey_c in zip(self.corridor.s, self.corridor.center_path_ey):
                    cx, cy = self.ref.se2xy(float(s_val), float(ey_c))
                    dp_pts.append(carla.Location(cx, cy, z0))
                for p0, p1 in zip(dp_pts[:-1], dp_pts[1:]):
                    dbg.draw_line(p0, p1, thickness=0.10, color=COL_DP, life_time=self.DEBUG_LIFE_TIME)

    # ------------------------------------------------------------
    # 车辆模型
    # ------------------------------------------------------------
    def vehicle_model_frenet(self, x, u):
        vx, ey, yaw_err, s = x
        accel, delta = u
        s_dot = vx * math.cos(yaw_err)
        k_ref = float(self.ref.kappa_at_s(float(s))) if self.ref is not None else 0.0
        ey_dot = vx * math.sin(yaw_err)
        yaw_err_dot = vx * math.tan(delta) / max(1e-6, self.WHEELBASE) - k_ref * s_dot
        vx_dot = accel
        return np.array([vx_dot, ey_dot, yaw_err_dot, s_dot], dtype=float)

    # ------------------------------------------------------------
    # NMPC 控制 (修复：死锁救援 + 起点边界放宽)
    # ------------------------------------------------------------
    def compute_control(self, ego: carla.Actor, dt: float = 0.05) -> Tuple[float, float, float, Dict[str, Any]]:
        """最简控制：只跟随黄色线（中心线），不做优化"""
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = float(math.hypot(vel.x, vel.y))

        if self.corridor is None or self.ref is None or len(self.corridor.s) < 3:
            return 0.0, 0.0, 0.4, {"opt_ok": False, "reason": "no_corridor"}

        self._ensure_vehicle_params(ego)
        max_steer = float(self._max_steer_rad) if self._max_steer_rad else float(math.radians(30.0))

        s_now, ey_now = self.ref.xy2se(tf.location.x, tf.location.y)
        if s_now is None:
            return 0.0, 0.0, 0.6, {"opt_ok": False, "reason": "ego_out_of_ref"}

        # 1) 取黄色线前方一个目标点（点跟）
        lookahead = 6.0  # 前视距离（m），可按需调整
        s_t = float(min(self.corridor.s[-1], s_now + lookahead))
        ey_t = float(np.interp(s_t, self.corridor.s, self.corridor.center_path_ey))
        x_t, y_t = self.ref.se2xy(s_t, ey_t)

        # 2) 转换到车体坐标系
        yaw = math.radians(tf.rotation.yaw)
        dx = x_t - tf.location.x
        dy = y_t - tf.location.y
        x_local = math.cos(yaw) * dx + math.sin(yaw) * dy
        y_local = -math.sin(yaw) * dx + math.cos(yaw) * dy
        Ld = max(0.1, math.hypot(x_local, y_local))

        # 3) Pure Pursuit 计算转角
        alpha = math.atan2(y_local, x_local)
        delta = math.atan2(2.0 * self.WHEELBASE * math.sin(alpha), Ld)
        steer = float(np.clip(delta / max_steer, -1.0, 1.0))

        # 4) 简单速度控制
        v_ref = float(self.v_ref_base)
        a_cmd = 0.35 * (v_ref - speed)
        if a_cmd >= 0:
            throttle = float(np.clip(a_cmd, 0.0, 0.8))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-a_cmd, 0.0, 0.4))

        # ✅ 兼容日志字段（避免 KeyError）
        lo = float(np.interp(s_now, self.corridor.s, self.corridor.bound_lower_safe))
        up = float(np.interp(s_now, self.corridor.s, self.corridor.bound_upper_safe))
        dbg = {
            "s": float(s_now),
            "ey": float(ey_now),
            "lo": float(lo),
            "up": float(up),
            "width": float(up - lo),
            "v": float(speed),
            "v_ref": float(v_ref),
            "steer": float(steer),
            "opt_ok": True,
            "lc": getattr(self, "_lc_state", ""),
            "lane": "",
        }
        return throttle, steer, brake, dbg


# ============================================================
def main(scenario_type: str = "cones"):
    print(f"\n{'=' * 60}")
    print("  Rule-Based Planner 场景测试")
    print(f"  场景类型: {scenario_type}")
    print(f"{'=' * 60}\n")

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    amap = world.get_map()

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
                front_speed_diff_pct=85.0,
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

        planner = RuleBasedPlanner(amap, v_ref_base=6.0)

        # ✅ [新增] 根据场景设置策略
        if scenario_type == "cones":
            planner.enable_auto_lane_change = False
            planner.enable_overtake = False
        elif scenario_type == "trimma":
            planner.enable_auto_lane_change = True
            planner.enable_overtake = True
        elif scenario_type == "construction":
            planner.enable_auto_lane_change = True
            planner.enable_overtake = False
        else:
            planner.enable_auto_lane_change = True

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
                obs, _ = env.step()

            if dbg and frame % 10 == 0:
                if dbg.get("opt_ok", False):
                    print(f"[CTRL] s={dbg['s']:.1f} ey={dbg['ey']:.2f} | lo={dbg['lo']:.2f} up={dbg['up']:.2f} "
                          f"w={dbg['width']:.2f} | v={dbg['v']:.2f}->{dbg['v_ref']:.2f} "
                          f"| steer={dbg.get('steer', 0.0):.2f} "
                          f"| dp_ok={dbg.get('dp_ok', False)} opt_ok=True | lc={dbg.get('lc', '')} lane={dbg.get('lane', '')}")
                else:
                    print(
                        f"[CTRL-FAIL] reason={dbg.get('reason', '')} msg={dbg.get('opt_msg', '')} lc={dbg.get('lc', '')}")

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
