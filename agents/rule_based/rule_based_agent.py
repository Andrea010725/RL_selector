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
        initial_upper_ey, initial_lower_ey = np.zeros(num_valid_s), np.zeros(num_valid_s)
        HIGH_COST_THRESHOLD = W_LANE
        for i in range(num_valid_s):
            center_idx = np.argmin(np.abs(ey_nodes - optimal_path_ey[i]))
            cost_slice = cost_map[i, :]
            left_idx, right_idx = center_idx, center_idx
            while left_idx + 1 < num_ey and cost_slice[left_idx + 1] < HIGH_COST_THRESHOLD: left_idx += 1
            initial_upper_ey[i] = ey_nodes[left_idx]
            while right_idx - 1 >= 0 and cost_slice[right_idx - 1] < HIGH_COST_THRESHOLD: right_idx -= 1
            initial_lower_ey[i] = ey_nodes[right_idx]

        corridor_upper_ey = initial_upper_ey.copy()
        corridor_lower_ey = initial_lower_ey.copy()

        #
        # corridor_width = initial_upper_ey - initial_lower_ey
        # blocked_indices = np.where(corridor_width < PASSABLE_WIDTH_THRESHOLD)[0]
        # is_blocked = len(blocked_indices) > 0
        #
        # if is_blocked:
        #     blockage_start_idx = blocked_indices[0] if len(blocked_indices) > 0 else 0
        #     print(f"[决策] 在索引 {blockage_start_idx} 处发现阻塞。检查相邻车道...")
        #     amap = world.get_map()
        #     s_of_blockage = s_nodes[blockage_start_idx]
        #     x_b, y_b = self.ref.se2xy(s_of_blockage, 0.0)
        #     decision_wp = amap.get_waypoint(carla.Location(x=x_b, y=y_b), project_to_road=True,
        #                                     lane_type=carla.LaneType.Driving)
        #
        #     can_go_left, can_go_right = False, False
        #
        #     # 首先获取当前车道的 lane_id
        #     current_lane_id = decision_wp.lane_id
        #
        #     # --- 检查左侧车道 ---
        #     import ipdb
        #     left_lane = decision_wp.get_left_lane()
        #     # 确保左侧车道存在，并且是可行驶的
        #     if left_lane and left_lane.lane_type == carla.LaneType.Driving:
        #         # 新的判断逻辑：通过 lane_id 的符号判断方向是否一致
        #         # 同向行驶的车道 lane_id 符号相同，因此它们的乘积必然大于0
        #         if current_lane_id * left_lane.lane_id > 0:
        #             # 检查路网信息是否允许向左变
        #              can_go_left = True
        #
        #     # --- 检查右侧车道 ---
        #     right_lane = decision_wp.get_right_lane()
        #     # 确保右侧车道存在，并且是可行驶的
        #     if right_lane and right_lane.lane_type == carla.LaneType.Driving:
        #         # 使用同样的逻辑判断右侧车道
        #         if current_lane_id * right_lane.lane_id > 0:
        #             # 检查路网信息是否允许向右变道
        #             can_go_right = True
        #
        #     print(f"[决策] 判断结果: can_go_left={can_go_left}, can_go_right={can_go_right}")
        #     if can_go_right:
        #         print("[决策] 右侧车道可用，按最终规则重构走廊！")
        #         for i in range(blockage_start_idx, num_valid_s):
        #             s_idx_ref = np.argmin(np.abs(self.ref.s - s_nodes[i]))
        #             waypoint = self.ref.wps[s_idx_ref]
        #
        #             cost_slice = cost_map[i, :]
        #             obstacle_indices = np.where(np.isinf(cost_slice))[0]
        #             obstacle_ey_values = ey_nodes[obstacle_indices]
        #             # ########## 核心修正点 1 ##########
        #             # 寻找位于车道中心线（ey=0）右侧的障碍物
        #             right_lane_obstacles = obstacle_ey_values[obstacle_ey_values > 0]
        #             if len(right_lane_obstacles) > 0:
        #                 corridor_lower_ey[i] = np.max(right_lane_obstacles)
        #             else:
        #                 corridor_lower_ey[i] = initial_lower_ey[i]
        #
        #             right_lane_wp = waypoint.get_right_lane()
        #             if right_lane_wp:
        #             #     corridor_upper_ey[i] = (waypoint.lane_width * 0.5 + right_lane_wp.lane_width)
        #             # else:
        #                 corridor_upper_ey[i] = initial_upper_ey[i]  # Fallback
        #
        #             for i in range(1, num_valid_s):
        #                 corridor_lower_ey[i] = max(corridor_lower_ey[i - 1], corridor_lower_ey[i])
        #
        #     elif can_go_left:
        #         print("[决策] 左侧车道可用，按最终规则重构走廊！")
        #         for i in range(blockage_start_idx, num_valid_s):
        #             cost_slice = cost_map[i, :]
        #             obstacle_indices = np.where(np.isinf(cost_slice))[0]
        #             obstacle_ey_values = ey_nodes[obstacle_indices]
        #             # ########## 核心修正点 2 ##########
        #             s_idx_ref = np.argmin(np.abs(self.ref.s - s_nodes[i]))
        #             waypoint = self.ref.wps[s_idx_ref]
        #             import ipdb
        #             # ipdb.set_trace()
        #             left_lane_wp = waypoint.get_left_lane()
        #             if left_lane_wp:
        #             #     corridor_lower_ey[i] = -(waypoint.lane_width * 0.5 + left_lane_wp.lane_width)
        #             # else:
        #                 corridor_lower_ey[i] = initial_lower_ey[i]  # Fallback
        #
        #             # 寻找位于车道中心线（ey=0）左侧的障碍物
        #             left_lane_obstacles = obstacle_ey_values[obstacle_ey_values < 0]
        #             if len(left_lane_obstacles) > 0:
        #                 corridor_upper_ey[i] = np.max(left_lane_obstacles)
        #             else:
        #                 corridor_upper_ey[i] = initial_upper_ey[i]
        #
        #             for i in range(1, num_valid_s):
        #                     corridor_upper_ey[i] = min(corridor_upper_ey[i - 1], corridor_upper_ey[i])

        # if not is_blocked:
        # for i in range(1, num_valid_s):
        #         corridor_upper_ey[i] = min(corridor_upper_ey[i - 1], corridor_upper_ey[i])
        #         corridor_lower_ey[i] = max(corridor_lower_ey[i - 1], corridor_lower_ey[i])

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

        # 1. 先生成自车，并获取其准确的初始路点
        ego, ego_wp = spawn_ego_upstream_lane_center(env)

        # 如果生成失败，ego_wp 可能是 None，需要处理
        if ego_wp is None:
            # 如果首选方案失败，后备方案返回的 ego_wp 是 None
            # 我们需要根据最终的 ego 位置重新获取一次
            ego_wp = env.world.get_map().get_waypoint(ego.get_location(),
                                                      project_to_road=True,
                                                      lane_type=carla.LaneType.Driving)
        if ego_wp is None:
            raise RuntimeError("无法为已生成的Ego车辆找到有效的路点。")

        # 2. 【核心修改】直接使用自车的路点 ego_wp 作为参考线的种子点
        print(f"[参考线生成] 使用自车所在位置的路点 {ego_wp.transform.location} 作为参考线起点。")
        amap = env.world.get_map()
        ref = LaneRef(amap, seed_wp=ego_wp, step=1.0, max_len=500.0)

        idp = 0.2 #  这里切换周围交通参与者的密度
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