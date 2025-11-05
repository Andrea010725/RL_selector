# agents/rule_based/agent.py
from __future__ import annotations
import math
import random
import sys
from types import SimpleNamespace
from scipy.optimize import minimize

from typing import List, Tuple

sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np
import ipdb

sys.path.append("/home/ajifang/czw/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.tools import SceneManager
from vis_debug import TelemetryLogger


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
        P = self.P
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
        # 注意：法向定义不要动（左法向）
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
                    print(f"    - 找到候选路点: {spawn_wp.transform.location}")

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
                    ego = world.try_spawn_actor(ego_bp, tf)
                    if ego:
                        env.set_ego(ego)
                        print(f"    ✅ [成功] 车辆已在后方 {back}米 处创建！")
                        print("--- [EGO 生成诊断 END] ---\n")
                        return ego, amap.get_waypoint(tf_location)
                    else:
                        print(f"    ❌ [失败] 生成失败。该位置可能被占用或无效。")
                else:
                    print(f"    - [跳过] 未能找到后方 {back}米 处的路点（可能道路太短）。")
        else:
            print("2. ❌ [失败] 在锥桶位置附近未能找到可行驶车道的路点。")
    else:
        print("1. ❌ [失败] 未能获取到第一个锥桶的位置。可能是场景生成失败。")

    # 后备方案
    print("\n[后备方案] 首选方案失败，现在尝试使用地图默认生成点...")
    spawns = amap.get_spawn_points()
    random.shuffle(spawns)
    for i, tf in enumerate(spawns[:10]):
        print(f"[后备方案] 尝试默认点 #{i + 1}...")
        tf.location.z += 0.20
        ego = world.try_spawn_actor(ego_bp, tf)
        if ego:
            env.set_ego(ego)
            print(f"    ✅ [成功] 车辆已在默认点创建！")
            print("--- [EGO 生成诊断 END] ---\n")
            return ego, None

    print("--- [EGO 生成诊断 END] ---\n")
    raise RuntimeError("所有方案都已尝试，未能生成EGO。请检查上面的诊断日志确定失败环节。")


# ====== 4) 可行驶区域（走廊）——地图车道线 ======
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
        wp = amap.get_waypoint(
            carla.Location(x=x, y=y, z=0.0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
        if wp is None:
            w = 3.5
        else:
            w = float(getattr(wp, "lane_width", 3.5)) or 3.5
        half = 0.5 * w
        left[i] = +half - lane_margin
        right[i] = -half + lane_margin
    return left, right


# ====== 4.x 方向一致性校准：判断“ey>0 是否真的是物理左侧” ======
def detect_ey_positive_is_left(ref: LaneRef, amap: carla.Map, s_probe: float) -> bool:
    """
    用“参考线处最近路点的左邻车道中心”来判定：
      若该点在 Frenet 下 ey>0 => ey 正方向 = 物理左
      若 ey<0 => ey 正方向 = 物理右（需要做标签翻转）
    """
    # 找 ref 中最靠近 s_probe 的 waypoint
    idx = int(np.argmin(np.abs(ref.s - s_probe)))
    wp_center = ref.wps[idx]

    # --- 方案一：优先使用 get_left_lane (最可靠) ---
    try:
        wp_left = wp_center.get_left_lane()
    except Exception:
        wp_left = None

    if wp_left and wp_left.lane_type == carla.LaneType.Driving:
        lx, ly = wp_left.transform.location.x, wp_left.transform.location.y
        _, eyL = ref.xy2se(lx, ly)
        ok = (eyL > 0.0)
        print(f"[EY侧校准] 左邻车道中心 ey={eyL:.3f} -> ey>0 是否物理左? {ok}")
        return ok

    # --- 方案二：如果方案一失败，使用更可靠的几何向量法进行兜底 ---
    sx, sy = ref.P[idx, 0], ref.P[idx, 1]
    tx, ty = ref.tang[idx]
    nx, ny = -ty, tx
    px, py = sx + 1.0 * nx, sy + 1.0 * ny
    probe_loc = carla.Location(x=px, y=py, z=wp_center.transform.location.z)

    right_vec = wp_center.transform.get_right_vector()
    probe_vec = probe_loc - wp_center.transform.location
    dot_product = probe_vec.x * right_vec.x + probe_vec.y * right_vec.y
    ok = (dot_product < 0.0)

    print(f"[EY侧校准-兜底] 使用向量法判定, 点积={dot_product:.3f} -> ey>0 是否物理左? {ok}")
    return ok


# ====== 4.1 [最终修正版] 仅用锥桶一侧夹逼 + 宽度不足时把“锥桶对向边界”向外扩 ======
def build_corridor_by_cones_one_side_only(
        world: carla.World,
        ego: carla.Actor,
        ref: LaneRef,
        s_ahead: float = 30.0,
        ds: float = 1.0,
        lane_margin: float = 0.20,
        cone_margin: float = 0.30,
        min_width: float = 1.8,
        force_normal_coord: bool = False
):
    amap = world.get_map()
    ego_tf = ego.get_transform()
    s0, _ = ref.xy2se(ego_tf.location.x, ego_tf.location.y)
    s_nodes = np.arange(s0, s0 + s_ahead, ds)

    # === 0) 侧向一致性校准 ===
    ey_pos_is_left = detect_ey_positive_is_left(ref, amap, s_probe=float(s_nodes[len(s_nodes) // 2]))
    if force_normal_coord:
        print("[手动校准] 已强制设定坐标系为正常模式 (ey>0 为物理左)")
        ey_pos_is_left = True

    # === 1) 基础边界（来自地图） ===
    left_ey_raw, right_ey_raw = lane_bounds_from_map(world, ref, s_nodes, lane_margin=lane_margin)

    # 将“数学左/右(ey符号)”映射为“物理左/右”标签
    if ey_pos_is_left:
        left_ey = left_ey_raw.copy()
        right_ey = right_ey_raw.copy()
    else:
        left_ey = right_ey_raw.copy()
        right_ey = left_ey_raw.copy()
    print(f"[侧向映射] ey>0 是否物理左: {ey_pos_is_left} | (left_ey,right_ey) 已按物理左右对齐")

    # === 2) 遍历“锥桶”，只夹逼对应一侧，并记录掩码（按物理左右） ===
    actors = world.get_actors()
    s_half_span_idx = max(1, int(0.5 * 1.2 / max(1e-6, ds)))

    cone_left_mask = np.zeros(len(s_nodes), dtype=bool)
    cone_right_mask = np.zeros(len(s_nodes), dtype=bool)

    print("\n[Corridor-By-Cones] 本周期锥桶侧判定与夹逼如下：")
    for a in actors:
        if getattr(a, "id", None) == ego.id: continue
        tname = getattr(a, "type_id", "").lower()
        if "spectator" in tname: continue
        if not tname.startswith("static.prop."): continue
        name = tname.split("static.prop.")[-1]
        if ("trafficcone" not in name) and ("traffic_cone" not in name) and ("trafficcone" not in tname): continue

        try:
            loc = a.get_location()
            s_obs_math, ey_obs_math = ref.xy2se(loc.x, loc.y)
        except Exception:
            continue

        if not (s_nodes[0] - 1.0 <= s_obs_math <= s_nodes[-1] + 1.0): continue

        is_left_cone = (ey_obs_math >= 0.0) if ey_pos_is_left else (ey_obs_math < 0.0)
        side_str = "LEFT(物理左)" if is_left_cone else "RIGHT(物理右)"

        s_idx_center = int(np.clip((s_obs_math - s_nodes[0]) / max(1e-6, ds), 0, len(s_nodes) - 1))
        lo_idx, hi_idx = max(0, s_idx_center - s_half_span_idx), min(len(s_nodes) - 1, s_idx_center + s_half_span_idx)
        print(
            f"  - cone@ s≈{s_obs_math:.2f}, ey_math≈{ey_obs_math:.2f}  -> 侧别(物理): {side_str} | 影响 s-idx [{lo_idx}, {hi_idx}]")

        target_ey = ey_obs_math - math.copysign(cone_margin, ey_obs_math)
        for k in range(lo_idx, hi_idx + 1):
            if is_left_cone:
                cone_left_mask[k] = True
                if ey_pos_is_left:
                    left_ey[k] = min(left_ey[k], target_ey)
                else:
                    left_ey[k] = max(left_ey[k], target_ey)
            else:
                cone_right_mask[k] = True
                if ey_pos_is_left:
                    right_ey[k] = max(right_ey[k], target_ey)
                else:
                    right_ey[k] = min(right_ey[k], target_ey)

    # === 3) 宽度不足则外扩：优先“锥桶的对向边界”（按物理左右） ===
    ego_width = ego.bounding_box.extent.y * 2.0
    width_thresh = max(min_width, ego_width + 0.30)

    def _drive_ok(wp: carla.Waypoint) -> bool:
        return (wp is not None) and (wp.lane_type == carla.LaneType.Driving)

    for i, s in enumerate(s_nodes):
        cur_w = abs(left_ey[i] - right_ey[i])
        if cur_w >= width_thresh: continue

        cx, cy = ref.se2xy(float(s), 0.0)
        center_location = carla.Location(x=cx, y=cy, z=0.0)
        center_wp = amap.get_waypoint(center_location, project_to_road=True, lane_type=carla.LaneType.Any)

        if center_wp is None:
            mid = 0.5 * (left_ey[i] + right_ey[i]);
            left_ey[i] = mid + 0.5 * width_thresh;
            right_ey[i] = mid - 0.5 * width_thresh
            print(f"  * 扩展兜底(center_wp None) @ s_idx={i} -> 对称扩到 {width_thresh:.2f}m")
            continue

        left_wp, right_wp = center_wp.get_left_lane(), center_wp.get_right_lane()
        left_ok, right_ok = _drive_ok(left_wp), _drive_ok(right_wp)

        cone_side = 'L' if (cone_left_mask[i] and not cone_right_mask[i]) else \
            'R' if (cone_right_mask[i] and not cone_left_mask[i]) else None

        expand_side = None
        if cone_side == 'L':
            if right_ok:
                expand_side = 'R'
            elif left_ok:
                expand_side = 'L'
        elif cone_side == 'R':
            if left_ok:
                expand_side = 'L'
            elif right_ok:
                expand_side = 'R'
        else:
            if left_ok:
                expand_side = 'L'
            elif right_ok:
                expand_side = 'R'

        if expand_side == 'L':
            adj_w = float(getattr(left_wp, "lane_width", 3.5)) or 3.5
            left_ey[i] += adj_w if ey_pos_is_left else -adj_w
            print(f"  * 宽度不足@ s_idx={i}（锥桶侧={cone_side}）-> 向左扩 {adj_w:.2f}m")
        elif expand_side == 'R':
            adj_w = float(getattr(right_wp, "lane_width", 3.5)) or 3.5
            right_ey[i] += -adj_w if ey_pos_is_left else adj_w
            print(f"  * 宽度不足@ s_idx={i}（锥桶侧={cone_side}）-> 向右扩 {adj_w:.2f}m")
        else:
            mid = 0.5 * (left_ey[i] + right_ey[i]);
            left_ey[i] = mid + 0.5 * width_thresh;
            right_ey[i] = mid - 0.5 * width_thresh
            print(f"  * 宽度不足@ s_idx={i}（锥桶侧={cone_side}）-> 两侧不可扩，对称扩到 {width_thresh:.2f}m")

        cur_w = abs(left_ey[i] - right_ey[i])
        if cur_w < width_thresh:
            mid = 0.5 * (left_ey[i] + right_ey[i]);
            left_ey[i] = mid + 0.5 * width_thresh;
            right_ey[i] = mid - 0.5 * width_thresh
            print(f"    - 二次兜底 @ s_idx={i} -> 对称扩到 {width_thresh:.2f}m")

    # === 4) 可视化点（物理左=upper，物理右=lower） ===
    upper_pts_world, lower_pts_world = [], []
    for s_val, up_ey, lo_ey in zip(s_nodes, left_ey, right_ey):
        ux, uy = ref.se2xy(s_val, up_ey)
        lx, ly = ref.se2xy(s_val, lo_ey)
        upper_pts_world.append(carla.Location(x=ux, y=uy))
        lower_pts_world.append(carla.Location(x=lx, y=ly))

    return s_nodes, left_ey, right_ey, upper_pts_world, lower_pts_world


def _col(r, g, b): return carla.Color(int(r), int(g), int(b))


# 参考线与中心线
COL_REF = _col(200, 200, 200)  # 参考线：浅灰
COL_CENTER = _col(255, 255, 255)  # 中心线：白

# 左/右边界：颜色区分明显（物理左=洋红，物理右=亮绿）
COL_LEFT = _col(255, 0, 255)  # 物理左（upper）
COL_RIGHT = _col(50, 220, 50)  # 物理右（lower）


# ====== 5) 规则型 Planner（中线跟随 + 边界保护 + 简易速度）======
class RuleBasedPlanner:
    def __init__(self, ref: LaneRef, v_ref_base: float = 12.0):
        self.ref = ref
        self.v_ref_base = float(v_ref_base)
        self.corridor = None
        self._prev_delta = 0.0
        self._prev_ax = 0.0

        # ---- Debug draw toggles ----
        self.DRAW_REF_LINE = True
        self.DRAW_CORRIDOR_EDGES = True
        self.DRAW_DP_CENTER = True
        self.DRAW_OBS_PRED_POINTS = False  # 关闭障碍点可视化
        self.DEBUG_LIFE_TIME = 0.8

    def update_corridor_simplified(
            self, world, ego,
            s_ahead=30.0, ds=1.0,
            ey_range=8.0, dey=0.15,
            horizon_T=2.0, dt=0.2,
            debug_draw=True,
            force_normal_coord: bool = False
    ):
        """
        极简版：不做 DP。
        规则（以物理左/右为准）：
          1) 只按“锥桶所在侧”向内夹逼该侧边界；另一侧不动；
          2) 若局部宽度不足，优先将“锥桶的对向边界向外扩”一个相邻 Driving 车道宽；
          3) 可视化：物理左(洋红)、物理右(亮绿)、中心(白)；无数值标注。
        """
        if not ego:
            self.corridor = None
            return

        s_nodes, left_ey, right_ey, up_pts, lo_pts = build_corridor_by_cones_one_side_only(
            world=world,
            ego=ego,
            ref=self.ref,
            s_ahead=s_ahead,
            ds=ds,
            lane_margin=0.20,
            cone_margin=0.30,
            min_width=1.8,
            force_normal_coord=force_normal_coord
        )

        # 中心线 = 左右边界中点（物理左右）
        center_ey = 0.5 * (left_ey + right_ey)
        self.corridor = SimpleNamespace(
            s=s_nodes,
            upper=left_ey,  # 物理左
            lower=right_ey,  # 物理右
            upper_pts_world=up_pts,
            lower_pts_world=lo_pts,
            center_path_ey=center_ey,
        )

        # 可视化：只画线
        if debug_draw and self.corridor:
            dbg = world.debug
            z0 = ego.get_location().z + 0.35

            # 参考线
            if self.DRAW_REF_LINE:
                step_idx = max(1, int(3.0 / max(1e-6, self.ref.step)))
                for p0, p1 in zip(self.ref.P[:-1:step_idx], self.ref.P[1::step_idx]):
                    dbg.draw_line(
                        carla.Location(p0[0], p0[1], z0),
                        carla.Location(p1[0], p1[1], z0),
                        thickness=0.06, color=COL_REF, life_time=self.DEBUG_LIFE_TIME
                    )

            # 左右边界（物理左=洋红，物理右=绿）
            if self.DRAW_CORRIDOR_EDGES:
                up_pts = self.corridor.upper_pts_world  # 物理左
                lo_pts = self.corridor.lower_pts_world  # 物理右
                for i in range(0, len(up_pts) - 1):
                    dbg.draw_line(
                        carla.Location(up_pts[i].x, up_pts[i].y, z0),
                        carla.Location(up_pts[i + 1].x, up_pts[i + 1].y, z0),
                        thickness=0.12, color=COL_LEFT, life_time=self.DEBUG_LIFE_TIME
                    )
                    dbg.draw_line(
                        carla.Location(lo_pts[i].x, lo_pts[i].y, z0),
                        carla.Location(lo_pts[i + 1].x, lo_pts[i + 1].y, z0),
                        thickness=0.12, color=COL_RIGHT, life_time=self.DEBUG_LIFE_TIME
                    )

            # 中心线（取左右中点）
            if self.DRAW_DP_CENTER:
                center_pts = []
                for s_val, ey_c in zip(self.corridor.s, self.corridor.center_path_ey):
                    cx, cy = self.ref.se2xy(s_val, ey_c)
                    center_pts.append(carla.Location(cx, cy, z0))
                for p0, p1 in zip(center_pts[:-1], center_pts[1:]):
                    dbg.draw_line(p0, p1, thickness=0.08, color=COL_CENTER, life_time=self.DEBUG_LIFE_TIME)

    def vehicle_model_frenet(self, x, u, L=2.5):
        """
        Frenet坐标系下的车辆动力学模型.
        x: [vx, ey, yaw_err, s]
        u: [accel, delta]
        """
        vx, ey, yaw_err, s = x
        accel, delta = u
        ey_dot = vx * math.sin(yaw_err)
        yaw_err_dot = vx * math.tan(delta) / L
        vx_dot = accel
        s_dot = vx * math.cos(yaw_err)
        return np.array([vx_dot, ey_dot, yaw_err_dot, s_dot])

    def compute_control(self, ego: carla.Actor, dt: float = 0.05):
        """
        横向：MPC控制，追踪动态的走廊中线
        纵向：简单的P控制器，以基础参考速度为目标
        """
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = float(math.hypot(vel.x, vel.y))
        x, y = tf.location.x, tf.location.y

        if self.corridor is None or len(self.corridor.s) < 3:
            return 0.0, 0.0, 1.0, {}  # 紧急刹车

        H = 10  # 预测时域
        L = ego.bounding_box.extent.x * 2.0

        s_now, ey_now = self.ref.xy2se(x, y)
        ego_yaw_rad = math.radians(tf.rotation.yaw)

        # 更精确的航向角误差
        s_idx = np.searchsorted(self.ref.s, s_now)
        s_idx = min(s_idx, len(self.ref.tang) - 1)
        ref_yaw_rad = math.atan2(self.ref.tang[s_idx, 1], self.ref.tang[s_idx, 0])
        yaw_err_now = ego_yaw_rad - ref_yaw_rad
        # Normalize error to [-pi, pi]
        yaw_err_now = (yaw_err_now + np.pi) % (2 * np.pi) - np.pi

        x0 = np.array([speed, ey_now, yaw_err_now, s_now])

        def objective_function(u):
            u = u.reshape((H, 2))
            W_CONTROL = 0.1
            W_CONTROL_RATE = 0.1
            W_EY = 10.0
            W_SPEED = 0.5

            cost_control = np.sum(u[:, 0] ** 2) + np.sum(u[:, 1] ** 2)
            cost_control_rate = np.sum(np.diff(u[:, 0]) ** 2) + np.sum(np.diff(u[:, 1]) ** 2)

            x_pred = x0.copy()
            cost_ey_tracking = 0.0
            cost_speed_tracking = 0.0

            for k in range(H):
                x_pred += self.vehicle_model_frenet(x_pred, u[k], L) * dt

                s_pred = x_pred[3]
                ey_target = np.interp(s_pred, self.corridor.s, self.corridor.center_path_ey)
                ey_error = x_pred[1] - ey_target
                cost_ey_tracking += ey_error ** 2

                speed_error = self.v_ref_base - x_pred[0]
                cost_speed_tracking += speed_error ** 2

            total_cost = (cost_control * W_CONTROL +
                          cost_control_rate * W_CONTROL_RATE +
                          cost_ey_tracking * W_EY +
                          cost_speed_tracking * W_SPEED)
            return total_cost

        cons = []

        def get_bounds_at_s(s, corridor):
            upper = np.interp(s, corridor.s, corridor.upper)
            lower = np.interp(s, corridor.s, corridor.lower)
            return upper, lower

        for k in range(H):
            def upper_constraint(u, k=k):
                u = u.reshape((H, 2))
                x_pred = x0.copy()
                for i in range(k + 1):
                    x_pred += self.vehicle_model_frenet(x_pred, u[i], L) * dt
                s_pred, ey_pred = x_pred[3], x_pred[1]
                upper, _ = get_bounds_at_s(s_pred, self.corridor)
                return upper - ey_pred

            def lower_constraint(u, k=k):
                u = u.reshape((H, 2))
                x_pred = x0.copy()
                for i in range(k + 1):
                    x_pred += self.vehicle_model_frenet(x_pred, u[i], L) * dt
                s_pred, ey_pred = x_pred[3], x_pred[1]
                _, lower = get_bounds_at_s(s_pred, self.corridor)
                return ey_pred - lower

            cons.append({'type': 'ineq', 'fun': upper_constraint})
            cons.append({'type': 'ineq', 'fun': lower_constraint})

        accel_min, accel_max = -5.0, 3.0
        delta_min, delta_max = -math.radians(30), math.radians(30)
        bounds = [(accel_min, accel_max), (delta_min, delta_max)] * H

        u_initial_guess = np.zeros(2 * H)
        result = minimize(objective_function, u_initial_guess, bounds=bounds, constraints=cons, method='SLSQP')

        optimal_u = result.x.reshape((H, 2))
        optimal_accel = optimal_u[0, 0]
        optimal_delta = optimal_u[0, 1]

        if optimal_accel > 0:
            throttle = float(np.clip(optimal_accel / accel_max, 0, 1))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(optimal_accel / accel_min, 0, 1))

        steer = float(np.clip(optimal_delta / delta_max, -1, 1))

        s_idx = np.argmin(np.abs(self.corridor.s - s_now))
        dbg_info = {
            's': s_now, 'ey': ey_now,
            'lo': self.corridor.lower[s_idx], 'up': self.corridor.upper[s_idx],
            'width': abs(self.corridor.upper[s_idx] - self.corridor.lower[s_idx]),
            'v': speed, 'v_ref': self.v_ref_base,
            'delta': optimal_delta, 'steer': steer,
            'throttle': throttle, 'brake': brake
        }
        return throttle, steer, brake, dbg_info


def set_spectator_above_start_point(
        world: carla.World,
        start_transform: carla.Transform,
        height: float = 35.0,
        distance_behind: float = 30.0,
        pitch: float = -45.0
):
    spectator = world.get_spectator()
    forward_vector = start_transform.get_forward_vector()
    camera_location = (
            start_transform.location
            - forward_vector * distance_behind
            + carla.Location(z=height)
    )
    camera_rotation = carla.Rotation(
        pitch=pitch,
        yaw=start_transform.rotation.yaw,
        roll=0.0
    )
    final_transform = carla.Transform(camera_location, camera_rotation)
    spectator.set_transform(final_transform)
    print(f"[Spectator] 观察者视角已固定在 {camera_location}")


# ====== 6) 主程序 ======
def main():
    FORCE_COORDINATE_SYSTEM_NORMAL = False  # 如果发现方向反了，可以改成False或True来手动校正

    env = HighwayEnv(host="127.0.0.1", port=2000, sync=True, fixed_dt=0.05).connect()
    logger = None
    try:
        env.setup_scene(
            num_cones=5, step_forward=3.0, step_right=0.35,
            z_offset=0.0, min_gap_from_junction=15.0,
            grid=5.0, set_spectator=True
        )

        ego, ego_wp = spawn_ego_upstream_lane_center(env)
        if ego_wp is None:
            raise RuntimeError("无法为已生成的Ego车辆找到有效的路点。")
        set_spectator_above_start_point(env.world, ego_wp.transform)

        print(f"[参考线生成] 使用自车所在位置的路点 {ego_wp.transform.location} 作为参考线起点。")
        amap = env.world.get_map()
        ref = LaneRef(amap, seed_wp=ego_wp, step=1.0, max_len=500.0)

        idp = 0.0
        scenemanager = SceneManager(ego_wp, idp)
        scenemanager.gen_traffic_flow(env.world, ego_wp)

        planner = RuleBasedPlanner(ref, v_ref_base=12.0)
        logger = TelemetryLogger(out_dir="logs_rule_based")

        dt = 0.05
        frame = 0

        while True:
            # 1) 更新走廊
            planner.update_corridor_simplified(env.world, ego, force_normal_coord=FORCE_COORDINATE_SYSTEM_NORMAL)

            # 2) 控制
            throttle, steer, brake, dbg = planner.compute_control(ego, dt=dt)

            # 3) 执行
            env.apply_control(throttle=throttle, steer=steer, brake=brake)

            # 4) 仿真步进 & 可视化
            obs, _ = env.step()
            if frame % 2 == 0:
                tf = ego.get_transform()
                # draw_ego_marker(env.world, tf.location.x, tf.location.y)

            # 5) 打印 & 记录
            if dbg and frame % 10 == 0:
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