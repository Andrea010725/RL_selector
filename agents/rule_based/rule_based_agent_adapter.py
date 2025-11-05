# my_agents/rule_based/agent.py
from __future__ import annotations
import math
from types import SimpleNamespace
from typing import Dict, Any, Tuple, Optional

import numpy as np
from scipy.optimize import minimize

import carla  # 由外部环境注入 PythonAPI 路径；此处不做 sys.path 修改

LowLevelAction = Tuple[float, float, float]  # (throttle, brake, steer)


# ========= 1) LaneRef：沿同一条驾驶车道采样，提供 xy<->(s,ey) =========
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
        self.wps = wps  # Waypoints，便于获取车道宽
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


# ========= 2) 规则型 Planner（DP 走廊 + 简化纵横向控制） =========
class RuleBasedPlanner:
    def __init__(self, ref: LaneRef, v_ref_base: float = 12.0):
        self.ref = ref
        self.v_ref_base = float(v_ref_base)
        self.corridor = None

    def update_corridor_simplified(
        self, world: carla.World, ego: carla.Actor,
        s_ahead=30.0, ds=1.0, ey_range=8.0, dey=0.15,
        horizon_T=2.0, dt=0.2, debug_draw=False
    ):
        """构建 Frenet 代价地图 + DP 求解中心线，并扫描得到左右边界。"""
        if not ego:
            self.corridor = None
            return

        ego_width = ego.bounding_box.extent.y * 2.0
        PASSABLE_WIDTH_THRESHOLD = ego_width + 0.6

        # 采样 s / ey
        x0, y0 = ego.get_location().x, ego.get_location().y
        s0, ey0 = self.ref.xy2se(x0, y0)
        s_nodes = np.arange(s0, s0 + s_ahead, ds)
        ey_nodes = np.arange(-ey_range, ey_range + dey, dey)
        num_s, num_ey = len(s_nodes), len(ey_nodes)

        # 代价地图：偏离中心惩罚 + 越界/逆向惩罚 + 障碍物占据
        cost_map = np.zeros((num_s, num_ey))
        W_LANE = 2e4
        W_OPPOSITE_LANE = 5e4
        W_OFFSET = 50.0
        cost_map += W_OFFSET * (ey_nodes ** 2)

        amap = world.get_map()
        for i, s in enumerate(s_nodes):
            s_idx_ref = int(np.argmin(np.abs(self.ref.s - s)))
            ref_wp = self.ref.wps[s_idx_ref]
            ref_lane_id = ref_wp.lane_id
            half_width = float(getattr(ref_wp, "lane_width", 3.5)) * 0.5
            for j, ey_val in enumerate(ey_nodes):
                if abs(ey_val) <= half_width:
                    continue
                x, y = self.ref.se2xy(s, ey_val)
                cell_wp = amap.get_waypoint(
                    carla.Location(x=x, y=y),
                    project_to_road=False,
                    lane_type=carla.LaneType.Any
                )
                if (cell_wp is None) or (cell_wp.lane_type != carla.LaneType.Driving):
                    cost_map[i, j] = W_LANE
                elif cell_wp.lane_id * ref_lane_id < 0:
                    cost_map[i, j] = W_OPPOSITE_LANE

        # 简化的障碍处理：静态/动态（常速外推）
        ego_loc = ego.get_location()
        OBSTACLE_RADIUS_M = 0.8
        actors = world.get_actors()
        for actor in actors:
            try:
                if actor.id == ego.id:
                    continue
                loc, type_id = actor.get_location(), actor.type_id
                if loc.distance(ego_loc) > s_ahead + 10:
                    continue
            except Exception:
                continue

            try:
                if type_id.startswith("vehicle.") or type_id.startswith("walker."):
                    vel = actor.get_velocity()
                    speed = math.hypot(vel.x, vel.y)
                    # 动态：预测若太慢则当静态处理
                    if speed < 0.5:
                        s, ey = self.ref.xy2se(loc.x, loc.y)
                        s_idx = int((s - s0) / ds)
                        if 0 <= s_idx < num_s:
                            idx = np.where(np.abs(ey_nodes - ey) < OBSTACLE_RADIUS_M + 1.0)[0]
                            cost_map[s_idx, idx] = float("inf")
                        continue
                    start_wp = amap.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
                    if not start_wp:
                        continue
                    for t in np.arange(0.0, 2.0, 0.2):
                        dist = speed * t
                        fut = start_wp.next(dist)
                        if not fut:
                            break
                        fut_loc = fut[0].transform.location
                        s, ey = self.ref.xy2se(fut_loc.x, fut_loc.y)
                        s_idx = int((s - s0) / ds)
                        if 0 <= s_idx < num_s:
                            idx = np.where(np.abs(ey_nodes - ey) < OBSTACLE_RADIUS_M)[0]
                            cost_map[s_idx, idx] = float("inf")
                else:
                    # 静态
                    s, ey = self.ref.xy2se(loc.x, loc.y)
                    s_idx = int((s - s0) / ds)
                    if 0 <= s_idx < num_s:
                        idx = np.where(np.abs(ey_nodes - ey) < OBSTACLE_RADIUS_M)[0]
                        cost_map[s_idx, idx] = float("inf")
            except Exception:
                continue

        # DP
        dp = np.full((num_s, num_ey), float("inf"))
        parent = np.zeros((num_s, num_ey), dtype=int)
        start_ey_idx = int(np.argmin(np.abs(ey_nodes - ey0)))
        dp[0, start_ey_idx] = 0.0
        W_STEER, W_JERK = 200.0, 500.0
        for i in range(1, num_s):
            for j in range(num_ey):
                if np.isinf(cost_map[i, j]):
                    continue
                for k in range(num_ey):
                    if np.isinf(dp[i - 1, k]):
                        continue
                    ey_c, ey_p = ey_nodes[j], ey_nodes[k]
                    steering_cost = (ey_c - ey_p) ** 2
                    jerk_cost = 0.0
                    if i > 1:
                        ey_gp = ey_nodes[parent[i - 1, k]]
                        jerk_cost = (ey_c - 2 * ey_p + ey_gp) ** 2
                    cand = dp[i - 1, k] + cost_map[i, j] + W_STEER * steering_cost + W_JERK * jerk_cost
                    if cand < dp[i, j]:
                        dp[i, j] = cand
                        parent[i, j] = k

        # 回溯路径
        opt_idx = np.zeros(num_s, dtype=int)
        if np.isinf(np.min(dp[-1, :])):
            # 走廊被完全阻塞：退回到最后一行可达处
            try:
                last = int(np.max(np.where(np.any(np.isfinite(dp), axis=1))[0]))
            except ValueError:
                last = 0
            opt_idx[last] = int(np.argmin(dp[last, :]))
            for i in range(last - 1, -1, -1):
                opt_idx[i] = parent[i + 1, opt_idx[i + 1]]
            opt_idx[last:] = opt_idx[last]
        else:
            opt_idx[-1] = int(np.argmin(dp[-1, :]))
            for i in range(num_s - 2, -1, -1):
                opt_idx[i] = parent[i + 1, opt_idx[i + 1]]
        path_ey = ey_nodes[opt_idx]

        # 扫描两侧边界
        HIGH_COST_THRESHOLD = 1e4
        upper_ey = np.zeros(num_s)
        lower_ey = np.zeros(num_s)
        for i in range(num_s):
            center_idx = int(np.argmin(np.abs(ey_nodes - path_ey[i])))
            row = cost_map[i, :]
            # 向“左侧”（ey增大）扫描
            u = center_idx
            while u + 1 < num_ey and row[u + 1] < HIGH_COST_THRESHOLD:
                u += 1
            # 向“右侧”（ey减小）扫描
            l = center_idx
            while l - 1 >= 0 and row[l - 1] < HIGH_COST_THRESHOLD:
                l -= 1
            upper_ey[i] = ey_nodes[u]
            lower_ey[i] = ey_nodes[l]

        # 打一点安全边距
        SAFETY_MARGIN_EY = 0.3
        self.corridor = SimpleNamespace(
            s=s_nodes,
            upper=upper_ey,
            lower=lower_ey,
            center_path_ey=path_ey,
            passable_width_threshold=PASSABLE_WIDTH_THRESHOLD,
            safety_margin=SAFETY_MARGIN_EY
        )

    # 简化的“自行车模型” + MPC 求第一个控制量
    def vehicle_model_frenet(self, x, u, L=2.5):
        vx, ey, yaw_err, s = x
        accel, delta = u
        ey_dot = vx * math.sin(yaw_err)
        yaw_err_dot = vx * math.tan(delta) / L
        vx_dot = accel
        s_dot = vx * math.cos(yaw_err)
        return np.array([vx_dot, ey_dot, yaw_err_dot, s_dot])

    def compute_control(self, ego: carla.Actor, dt: float = 0.05):
        """输出 (throttle, steer, brake, dbg) —— 注意返回顺序！"""
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = float(math.hypot(vel.x, vel.y))
        x, y = tf.location.x, tf.location.y

        if self.corridor is None or len(self.corridor.s) < 3:
            return 0.0, 0.0, 1.0, {}  # 紧急刹车

        H = 10
        L = ego.bounding_box.extent.x * 2.0

        s_now, ey_now = self.ref.xy2se(x, y)
        yaw_err_now = 0.0  # 简化
        x0 = np.array([speed, ey_now, yaw_err_now, s_now])

        def obj(u):
            u = u.reshape((H, 2))
            W_CONTROL = 0.1
            W_CONTROL_RATE = 0.1
            W_EY = 10.0
            W_SPEED = 0.5

            c_ctrl = np.sum(u[:, 0] ** 2) + np.sum(u[:, 1] ** 2)
            c_rate = np.sum(np.diff(u[:, 0]) ** 2) + np.sum(np.diff(u[:, 1]) ** 2)
            x_pred = x0.copy()
            c_ey = 0.0
            c_spd = 0.0
            for k in range(H):
                x_pred += self.vehicle_model_frenet(x_pred, u[k], L) * dt
                c_ey += x_pred[1] ** 2
                c_spd += (self.v_ref_base - x_pred[0]) ** 2
            return W_CONTROL * c_ctrl + W_CONTROL_RATE * c_rate + W_EY * c_ey + W_SPEED * c_spd

        cons = []
        def get_bounds_at_s(s):
            upper = np.interp(s, self.corridor.s, self.corridor.upper) - self.corridor.safety_margin
            lower = np.interp(s, self.corridor.s, self.corridor.lower) + self.corridor.safety_margin
            return upper, lower

        for k in range(H):
            def upper_c(u, k=k):
                u = u.reshape((H, 2))
                x_pred = x0.copy()
                for i in range(k + 1):
                    x_pred += self.vehicle_model_frenet(x_pred, u[i], L) * dt
                s_pred, ey_pred = x_pred[3], x_pred[1]
                up, _ = get_bounds_at_s(s_pred)
                return up - ey_pred
            def lower_c(u, k=k):
                u = u.reshape((H, 2))
                x_pred = x0.copy()
                for i in range(k + 1):
                    x_pred += self.vehicle_model_frenet(x_pred, u[i], L) * dt
                s_pred, ey_pred = x_pred[3], x_pred[1]
                _, lo = get_bounds_at_s(s_pred)
                return ey_pred - lo
            cons.append({'type': 'ineq', 'fun': upper_c})
            cons.append({'type': 'ineq', 'fun': lower_c})

        accel_min, accel_max = -5.0, 3.0
        delta_min, delta_max = -math.radians(30), math.radians(30)
        bounds = [(accel_min, accel_max), (delta_min, delta_max)] * H

        u0 = np.zeros(2 * H)
        result = minimize(obj, u0, bounds=bounds, constraints=cons, method="SLSQP")

        u_star = result.x.reshape((H, 2))
        ax, delta = float(u_star[0, 0]), float(u_star[0, 1])

        if ax > 0:
            throttle = float(np.clip(ax / accel_max, 0, 1))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(ax / accel_min, 0, 1))
        steer = float(np.clip(delta / delta_max, -1, 1))

        s_idx = int(np.argmin(np.abs(self.corridor.s - s_now)))
        dbg = {
            "s": s_now, "ey": ey_now,
            "lo": float(self.corridor.lower[s_idx]), "up": float(self.corridor.upper[s_idx]),
            "width": float(self.corridor.upper[s_idx] - self.corridor.lower[s_idx]),
            "v": speed, "v_ref": self.v_ref_base,
            "delta": delta, "steer": steer, "throttle": throttle, "brake": brake
        }
        # 注意此处返回顺序 (throttle, steer, brake, dbg)
        return throttle, steer, brake, dbg


# ========= 3) 适配器：供 router/env_wrapper 调用 =========
class RulePlannerAdapter:
    """无副作用的适配器：reset/attach_context/plan"""
    def __init__(self, v_ref_base: float = 12.0):
        self._world: Optional[carla.World] = None
        self._ego: Optional[carla.Actor] = None
        self._ref: Optional[LaneRef] = None
        self._planner: Optional[RuleBasedPlanner] = None
        self.v_ref_base = float(v_ref_base)

    def attach_context(self, world: carla.World, ego: carla.Actor, ref: LaneRef):
        self._world = world
        self._ego = ego
        self._ref = ref
        self._planner = RuleBasedPlanner(ref, v_ref_base=self.v_ref_base)

    def reset(self):
        if self._planner:
            self._planner.corridor = None

    def plan(self, obs: np.ndarray, info: Dict[str, Any] = None) -> LowLevelAction:
        assert self._world is not None and self._ego is not None and self._planner is not None, \
            "RulePlannerAdapter 未 attach_context()"
        self._planner.update_corridor_simplified(
            world=self._world, ego=self._ego,
            s_ahead=30.0, ds=1.0, ey_range=8.0, dey=0.15,
            horizon_T=2.0, dt=0.2, debug_draw=False
        )
        throttle, steer, brake, _ = self._planner.compute_control(self._ego, dt=0.05)
        # 与 CarlaEnv.step 的低层控制一致：(throttle, brake, steer)
        return float(throttle), float(brake), float(steer)
