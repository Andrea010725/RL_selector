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
            for back in [37.0, 38.0, 39.0]:  # 使用完整的距离列表以提高成功率
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

    def update_corridor_simplified(self, world, ego, s_ahead=30.0, ds=1.0, ey_range=8.0, dey=0.15, horizon_T=2.0,
                                   dt=0.2, debug_draw=True):
        """
        使用动态规划在Frenet坐标系下进行路径规划 (V9 - 修正重构逻辑中的障碍物判断基准)
        """
        if not ego:
            self.corridor = None
            return

        ego_width = ego.bounding_box.extent.y * 2.0
        PASSABLE_WIDTH_THRESHOLD = ego_width + 0.6
        s0, ey0 = self.ref.xy2se(ego.get_location().x, ego.get_location().y)
        s_nodes = np.arange(s0, s0 + s_ahead, ds)
        ey_nodes = np.arange(-ey_range, ey_range + dey, dey)
        num_s, num_ey = len(s_nodes), len(ey_nodes)

        # ================= 2. 构建代价地图 (Cost Map) =================
        cost_map = np.zeros((num_s, num_ey))
        W_LANE = 20000.0
        W_OPPOSITE_LANE = 50000.0
        W_OFFSET = 50
        offset_cost = W_OFFSET * (ey_nodes ** 2)
        cost_map += offset_cost
        amap = world.get_map()  # 提前获取地图对象

        for i, s in enumerate(s_nodes):
            s_idx_ref = np.argmin(np.abs(self.ref.s - s))
            ref_waypoint = self.ref.wps[s_idx_ref]
            ref_lane_id = ref_waypoint.lane_id
            half_width = ref_waypoint.lane_width * 0.5

            # 遍历该s值下的所有横向采样点ey
            for j, ey_val in enumerate(ey_nodes):
                # 1. 判断该(s, ey)点是否在当前参考车道内
                if abs(ey_val) <= half_width:
                    # 在当前车道内，不施加任何惩罚
                    continue

                # 2. 对于车道外的点，将其从 (s, ey) 转回世界坐标 (x, y)
                x, y = self.ref.se2xy(s, ey_val)

                # 3. 获取该世界坐标点对应的路点信息
                cell_waypoint = amap.get_waypoint(carla.Location(x=x, y=y),
                                                  project_to_road=False,
                                                  lane_type=carla.LaneType.Any)

                # 4. 根据路点信息施加惩罚
                if cell_waypoint is None or cell_waypoint.lane_type != carla.LaneType.Driving:
                    # 如果该点没有路点信息(在路外)，或者不是可行驶车道
                    cost_map[i, j] = W_LANE
                elif cell_waypoint.lane_id * ref_lane_id < 0:
                    import ipdb
                    # 如果该点是可行驶车道，但与参考车道方向相反
                    cost_map[i, j] = W_OPPOSITE_LANE

        actors = world.get_actors()
        ego_loc = ego.get_location()
        OBSTACLE_RADIUS_M = 0.8
        for actor in actors:
            if actor.id == ego.id or "spectator" in actor.type_id: continue
            try:
                loc, type_id = actor.get_location(), actor.type_id
                if loc.distance(ego_loc) > s_ahead + 10: continue
            except Exception:
                continue
            is_static_prop = type_id.startswith("static.prop.")
            if is_static_prop:
                try:
                    s, ey = self.ref.xy2se(loc.x, loc.y)
                    s_idx = int((s - s0) / ds)
                    if not (0 <= s_idx < num_s): continue
                    indices_to_penalize = np.where(np.abs(ey_nodes - ey) < OBSTACLE_RADIUS_M)[0]
                    cost_map[s_idx, indices_to_penalize] = float('inf')
                except IndexError:
                    continue

            # ==================== [新增代码 START] 对动态车辆进行轨迹预测 ====================
            elif type_id.startswith("vehicle."):
                # 定义预测参数
                PREDICTION_HORIZON_S = 2.0  # 向前预测 2.0 秒
                PREDICTION_STEP_S = 0.2  # 预测时间步长为 0.2 秒

                try:
                    # 1. 获取该车辆的当前速度和所在路点
                    vel = actor.get_velocity()
                    speed = math.sqrt(vel.x ** 2 + vel.y ** 2)

                    # 如果车辆几乎是静止的，按静态障碍物处理，简化计算
                    if speed < 0.5:
                        s, ey = self.ref.xy2se(loc.x, loc.y)
                        s_idx = int((s - s0) / ds)
                        if 0 <= s_idx < num_s:
                            indices = np.where(np.abs(ey_nodes - ey) < OBSTACLE_RADIUS_M + 1.0)[0]  # 给予更大半径
                            cost_map[s_idx, indices] = float('inf')
                        continue  # 处理下一个actor

                    amap = world.get_map()
                    start_wp = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if not start_wp:
                        continue

                    # 2. 循环预测未来时间点的路径点
                    for t in np.arange(0.0, PREDICTION_HORIZON_S, PREDICTION_STEP_S):
                        # 计算在该时间点，车辆沿其车道行驶的距离
                        dist = speed * t
                        # 使用CARLA API获取未来路点
                        # .next(distance) 会返回一个列表，包含沿车道中心线前方`distance`米处的路点
                        future_wps = start_wp.next(dist)
                        if not future_wps:
                            break  # 如果前方没有路了，就停止对该车的预测

                        future_wp = future_wps[0]
                        future_loc = future_wp.transform.location

                        # 3. 将预测到的未来位置点，在我们的代价地图上标记为障碍
                        s, ey = self.ref.xy2se(future_loc.x, future_loc.y)

                        s_idx = int((s - s0) / ds)
                        if not (0 <= s_idx < num_s):
                            continue  # 预测点超出了我们的规划范围

                        # 确定障碍物影响的ey范围，并设置为无穷大代价
                        # OBSTACLE_RADIUS_M 可以根据车辆宽度进行调整，这里沿用之前的设置
                        indices_to_penalize = np.where(np.abs(ey_nodes - ey) < OBSTACLE_RADIUS_M)[0]
                        cost_map[s_idx, indices_to_penalize] = float('inf')

                except Exception as e:
                    # 捕获可能的异常，例如xy2se转换失败，避免程序崩溃
                    # print(f"Warning: Failed to predict actor {actor.id}. Error: {e}")
                    continue


        # ================= 3. DP求解 =================
        dp_table = np.full((num_s, num_ey), float('inf'))
        parent_table = np.zeros((num_s, num_ey), dtype=int)
        start_ey_idx = np.argmin(np.abs(ey_nodes - ey0))
        dp_table[0, start_ey_idx] = 0
        W_STEER, W_JERK = 200.0, 500.0
        for i in range(1, num_s):
            for j in range(num_ey):
                if np.isinf(cost_map[i, j]): continue
                for k in range(num_ey):
                    if np.isinf(dp_table[i - 1, k]): continue
                    ey_curr, ey_prev = ey_nodes[j], ey_nodes[k]
                    steering_cost = (ey_curr - ey_prev) ** 2
                    jerk_cost = 0
                    if i > 1:
                        ey_grandparent = ey_nodes[parent_table[i - 1, k]]
                        jerk_cost = (ey_curr - 2 * ey_prev + ey_grandparent) ** 2
                    transition_cost = W_STEER * steering_cost + W_JERK * jerk_cost
                    total_cost = dp_table[i - 1, k] + cost_map[i, j] + transition_cost
                    if total_cost < dp_table[i, j]: dp_table[i, j] = total_cost; parent_table[i, j] = k

        # ================= 4. 回溯最优路径 =================
        optimal_path_indices = np.zeros(num_s, dtype=int)
        if np.isinf(np.min(dp_table[-1, :])):
            print("[DP Planner] 警告: 路径被完全阻塞!")
            try:
                last_s_idx = np.max(np.where(np.any(np.isfinite(dp_table), axis=1))[0])
            except ValueError:
                last_s_idx = 0
            optimal_path_indices[last_s_idx] = np.argmin(dp_table[last_s_idx, :])
            for i in range(last_s_idx - 1, -1, -1): optimal_path_indices[i] = parent_table[
                i + 1, optimal_path_indices[i + 1]]
            optimal_path_indices[last_s_idx:] = optimal_path_indices[last_s_idx]
        else:
            optimal_path_indices[-1] = np.argmin(dp_table[-1, :])
            for i in range(num_s - 2, -1, -1): optimal_path_indices[i] = parent_table[
                i + 1, optimal_path_indices[i + 1]]
        optimal_path_ey = ey_nodes[optimal_path_indices]
        num_valid_s = len(optimal_path_ey)

        # ================= 5. [最终修正] 提取、决策、并按明确规则重构走廊 =================
        final_upper_ey = np.zeros(num_valid_s)
        final_lower_ey = np.zeros(num_valid_s)

        # 设置一个合理的阈值，它必须低于 W_LANE 和 W_OPPOSITE_LANE
        # 这样扫描时才能在这些惩罚区前停下。'inf'成本的障碍物自然也会让它停下。
        HIGH_COST_THRESHOLD = 10000.0

        for i in range(num_valid_s):
            # optimal_path_ey[i] 是DP算法给出的在s_nodes[i]处的最佳ey位置
            center_ey = optimal_path_ey[i]
            center_idx = np.argmin(np.abs(ey_nodes - center_ey))
            cost_slice = cost_map[i, :]

            # 从最优路径点向左扫描边界
            upper_idx = center_idx
            while upper_idx + 1 < num_ey and cost_slice[upper_idx + 1] < HIGH_COST_THRESHOLD:
                upper_idx += 1
            final_upper_ey[i] = ey_nodes[upper_idx]

            # 从最优路径点向右扫描边界
            lower_idx = center_idx
            while lower_idx - 1 >= 0 and cost_slice[lower_idx - 1] < HIGH_COST_THRESHOLD:
                lower_idx -= 1
            final_lower_ey[i] = ey_nodes[lower_idx]

        # 直接将扫描结果作为最终走廊
        corridor_upper_ey = final_upper_ey
        corridor_lower_ey = final_lower_ey

        # ================= 6. 可视化与输出 =================
        corridor_s = s_nodes[:num_valid_s]
        upper_pts, lower_pts = [], []
        SAFETY_MARGIN_EY = 0.3
        for s_val, upper_ey, lower_ey in zip(corridor_s, corridor_upper_ey - SAFETY_MARGIN_EY,
                                             corridor_lower_ey + SAFETY_MARGIN_EY):
            ux, uy = self.ref.se2xy(s_val, upper_ey)
            lx, ly = self.ref.se2xy(s_val, lower_ey)
            upper_pts.append(carla.Location(x=ux, y=uy))
            lower_pts.append(carla.Location(x=lx, y=ly))

        self.corridor = SimpleNamespace(s=corridor_s, lower=corridor_lower_ey, upper=corridor_upper_ey,
                                        upper_pts_world=upper_pts, lower_pts_world=lower_pts,
                                        center_path_ey=optimal_path_ey)

        if debug_draw and self.corridor:
            dbg, life_time = world.debug, 0.2
            z_offset = ego.get_location().z + 0.2
            for i in range(len(upper_pts) - 1):
                width = upper_pts[i].distance(lower_pts[i])
                is_blocked_viz = width < PASSABLE_WIDTH_THRESHOLD
                color_upper = carla.Color(255, 0, 0) if is_blocked_viz else carla.Color(64, 255, 255)
                color_lower = carla.Color(255, 0, 0) if is_blocked_viz else carla.Color(255, 235, 64)
                p_upper_1 = carla.Location(upper_pts[i].x, upper_pts[i].y, z_offset)
                p_upper_2 = carla.Location(upper_pts[i + 1].x, upper_pts[i + 1].y, z_offset)
                p_lower_1 = carla.Location(lower_pts[i].x, lower_pts[i].y, z_offset)
                p_lower_2 = carla.Location(lower_pts[i + 1].x, lower_pts[i + 1].y, z_offset)
                dbg.draw_line(p_upper_1, p_upper_2, thickness=0.1, color=color_upper, life_time=life_time,
                               persistent_lines=False)
                dbg.draw_line(p_lower_1, p_lower_2, thickness=0.1, color=color_lower, life_time=life_time,
                               persistent_lines=False)

    # 搭建车辆模型
    def vehicle_model_frenet(self, x, u, L=2.5):
        """
        Frenet坐标系下的车辆动力学模型.

        :param x: 状态向量 [vx, ey, yaw_err, s]
                        vx: 纵向速度 (m/s)
                        ey: 横向偏移 (m)
                        yaw_err: 航向角误差 (rad)
                        s: 沿参考线的纵向距离 (m)
        :param u: 控制向量 [accel, delta]
                        accel: 加速度 (m/s^2)
                        delta: 前轮转角 (rad)
        :param L: 车辆轴距 (m)
        :return: 状态量的变化率 [vx_dot, ey_dot, yaw_err_dot, s_dot]
        """
        vx, ey, yaw_err, s = x
        accel, delta = u

        # 假设横向速度 vy 近似为 0，侧偏角 beta 通过 yaw_err 和 ey_dot 近似

        # 横向偏移的变化率
        ey_dot = vx * math.sin(yaw_err)

        # 航向角误差的变化率 (车辆的横摆角速度)
        # 经典自行车模型： omega = v * tan(delta) / L
        yaw_err_dot = vx * math.tan(delta) / L

        # 纵向速度的变化率
        vx_dot = accel

        # 纵向距离 s 的变化率
        s_dot = vx * math.cos(yaw_err)

        return np.array([vx_dot, ey_dot, yaw_err_dot, s_dot])

    def compute_control(self, ego: carla.Actor, dt: float = 0.05):
        """
        横向：纯追踪控制，取走廊中线作为目标路径
        纵向：简单的P控制器，以基础参考速度为目标
        """
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = float(math.hypot(vel.x, vel.y))
        x, y = tf.location.x, tf.location.y

        # --- 1. 检查走廊是否存在 ---
        if self.corridor is None or len(self.corridor.s) < 3:
            return 0.0, 0.0, 1.0, {}  # 紧急刹车

        # --- 2. 定义MPC参数 ---
        H = 10  # 预测时域 (Horizon), 例如10步，对应 10 * 0.1 = 1.0秒
        # 车辆轴距 (一个合理的默认值)
        L = ego.bounding_box.extent.x * 2.0

        # --- 3. 获取当前状态 (Frenet坐标系) ---
        s_now, ey_now = self.ref.xy2se(x, y)
        # TODO: yaw_err_now 需要根据车辆当前朝向和参考线朝向计算得到
        yaw_err_now = 0.0  # 简化处理，实际需要计算
        x0 = np.array([speed, ey_now, yaw_err_now, s_now])

        # --- 4. 定义目标/代价函数 ---
        def objective_function(u):
            u = u.reshape((H, 2))

            # 权重参数 (方便统一调整)
            W_CONTROL = 0.1
            W_CONTROL_RATE = 0.1
            W_EY = 10.0
            W_SPEED = 0.5  # 【新增】速度跟踪的权重

            # a) 控制量和变化率代价
            cost_control = np.sum(u[:, 0] ** 2) + np.sum(u[:, 1] ** 2)
            cost_control_rate = np.sum(np.diff(u[:, 0]) ** 2) + np.sum(np.diff(u[:, 1]) ** 2)

            # 【核心修正】在预测过程中累加每一步的代价
            x_pred = x0.copy()
            cost_ey_tracking = 0.0
            cost_speed_tracking = 0.0

            for k in range(H):
                # 预测下一个状态
                x_pred += self.vehicle_model_frenet(x_pred, u[k], L) * dt

                # 【新增】累加每一步的横向偏移代价
                cost_ey_tracking += x_pred[1] ** 2

                # 【新增】累加每一步的速度跟踪代价
                speed_error = self.v_ref_base - x_pred[0]
                cost_speed_tracking += speed_error ** 2

            # 最终总代价
            total_cost = (cost_control * W_CONTROL +
                          cost_control_rate * W_CONTROL_RATE +
                          cost_ey_tracking * W_EY +
                          cost_speed_tracking * W_SPEED)

            return total_cost

        # --- 5. 定义约束条件 ---
        cons = []
        # a) 边界约束
        for k in range(H):
            # 不等式约束 c(x) >= 0
            # upper_bound - ey >= 0  和  ey - lower_bound >= 0
            # 这里的 x_k 是第k步的预测状态，ey是x_k[1]
            # 我们需要一个函数来根据s值获取边界
            def get_bounds_at_s(s, corridor):
                upper = np.interp(s, corridor.s, corridor.upper)
                lower = np.interp(s, corridor.s, corridor.lower)
                return upper, lower

            # 由于约束函数需要以 (u) 为输入，我们需要在内部进行预测
            def upper_constraint(u, k):
                u = u.reshape((H, 2))
                x_pred = x0.copy()
                for i in range(k + 1):
                    x_pred += self.vehicle_model_frenet(x_pred, u[i], L) * dt
                s_pred, ey_pred = x_pred[3], x_pred[1]
                upper, _ = get_bounds_at_s(s_pred, self.corridor)
                return upper - ey_pred  # >= 0

            def lower_constraint(u, k):
                u = u.reshape((H, 2))
                x_pred = x0.copy()
                for i in range(k + 1):
                    x_pred += self.vehicle_model_frenet(x_pred, u[i], L) * dt
                s_pred, ey_pred = x_pred[3], x_pred[1]
                _, lower = get_bounds_at_s(s_pred, self.corridor)
                return ey_pred - lower  # >= 0

            cons.append({'type': 'ineq', 'fun': lambda u, k=k: upper_constraint(u, k)})
            cons.append({'type': 'ineq', 'fun': lambda u, k=k: lower_constraint(u, k)})

        # b) 控制量范围约束
        accel_min, accel_max = -5.0, 3.0
        delta_min, delta_max = -math.radians(30), math.radians(30)
        bounds = [(accel_min, accel_max), (delta_min, delta_max)] * H

        # --- 6. 求解优化问题 ---
        u_initial_guess = np.zeros(2 * H)  # 初始猜测
        result = minimize(objective_function, u_initial_guess, bounds=bounds, constraints=cons, method='SLSQP')

        # --- 7. 提取并应用第一个控制指令 ---
        optimal_u = result.x.reshape((H, 2))
        optimal_accel = optimal_u[0, 0]
        optimal_delta = optimal_u[0, 1]

        # --- 8. 将加速度和转角转换为油门和刹车 ---
        if optimal_accel > 0:
            throttle = float(np.clip(optimal_accel / accel_max, 0, 1))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(optimal_accel / accel_min, 0, 1))

        steer = float(np.clip(optimal_delta / delta_max, -1, 1))

        # (返回控制量和调试信息)
        s_idx = np.argmin(np.abs(self.corridor.s - s_now))
        dbg_info = {
            's': s_now, 'ey': ey_now,
            'lo': self.corridor.lower[s_idx], 'up': self.corridor.upper[s_idx],
            'width': self.corridor.upper[s_idx] - self.corridor.lower[s_idx],
            'v': speed, 'v_ref': self.v_ref_base,  # 使用基础速度作为参考
            'delta': optimal_delta, 'steer': steer,
            'throttle': throttle, 'brake': brake
        }
        return throttle, steer, brake, dbg_info


# ====== 6) 主程序 ======
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

        # 如果生成失败，ego_wp 可能是 None，需要处理
        # ego_wp = env.world.get_map().get_waypoint(ego.get_location(),
        #                                               project_to_road=True,
        #                                                       lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            raise RuntimeError("无法为已生成的Ego车辆找到有效的路点。")

        # 2. 【核心修改】直接使用自车的路点 ego_wp 作为参考线的种子点
        print(f"[参考线生成] 使用自车所在位置的路点 {ego_wp.transform.location} 作为参考线起点。")
        amap = env.world.get_map()
        ref = LaneRef(amap, seed_wp=ego_wp, step=1.0, max_len=500.0)

        idp = 0.0 #  这里切换周围交通参与者的密度
        scenemanager = SceneManager(ego_wp, idp)
        import ipdb
        # ipdb.set_trace()
        scenemanager.gen_traffic_flow(env.world, ego_wp)

        planner = RuleBasedPlanner(ref, v_ref_base=12.0)
        logger = TelemetryLogger(out_dir="logs_rule_based")

        dt = 0.05
        frame = 0

        # import ipdb
        # ipdb.set_trace()
        while True:
            # 1) 更新走廊（始终有线；有障碍则收紧）
            planner.update_corridor_simplified(env.world, ego)

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