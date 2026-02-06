# agents/rule_based/agent.py
from __future__ import annotations
import math
import sys
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Any, List

from scipy.optimize import minimize

sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np

# 工程依赖
sys.path.append("/home/ajifang/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.scenarios import JaywalkerScenario, TrimmaScenario, ConstructionLaneChangeScenario, ConesScenario

sys.path.append("/home/ajifang/Driveadapter_2/tools")
from custom_eval import TrafficFlowSpawner
from vis_debug import TelemetryLogger

# EVA Monitor
try:
    sys.path.append("/home/ajifang/RL_selector/agents/rule_based/")
    from eva_monitor import EvaMonitor
except ImportError:
    EvaMonitor = None
    print("[Warning] EvaMonitor not found.")


# ============================================================
# 颜色定义
# ============================================================
def _col(r, g, b): return carla.Color(int(r), int(g), int(b))


COL_REF = _col(200, 200, 200)
COL_DP = _col(255, 255, 0)


# ============================================================
# 1) LaneRef：参考线 (修复：起始阶段放宽检查，防止断线)
# ============================================================
class LaneRef:
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len: float = 200.0):
        pts, wps = [], []
        wp = seed_wp

        dist = 0.0
        pts.append((wp.transform.location.x, wp.transform.location.y))
        wps.append(wp)

        # ✅ [修复] 放宽角度阈值，防止一点点弯道就断开
        # 0.5 对应 60度，足够排除横向道路，但允许弯道
        COS_THRESH = 0.5

        while dist < max_len:
            nxts = wp.next(step)
            if not nxts: break

            best_wp = None
            max_dot = -1.0
            fwd = wp.transform.get_forward_vector()

            for n_cand in nxts:
                if n_cand.lane_type != carla.LaneType.Driving: continue

                vec = n_cand.transform.location - wp.transform.location
                norm = math.sqrt(vec.x ** 2 + vec.y ** 2 + vec.z ** 2)
                if norm < 1e-3: continue

                dot = (vec.x * fwd.x + vec.y * fwd.y + vec.z * fwd.z) / norm

                if dot > max_dot:
                    max_dot = dot
                    best_wp = n_cand

            # ✅ [修复] 起始 5 米内强制生成，无视角度检查
            # 确保参考线至少有一段，能让车跑起来
            is_start_phase = (dist < 5.0)

            if best_wp is None:
                break

            if (not is_start_phase) and (max_dot < COS_THRESH):
                # 只有在跑出一段距离后，才进行严格检查，防止拐到横向路
                break

            wp = best_wp
            pts.append((wp.transform.location.x, wp.transform.location.y))
            wps.append(wp)
            dist += step

        self.P = np.asarray(pts, dtype=float)

        # 数据结构构建
        if len(self.P) < 2:
            # 兜底：如果真的生成失败，造一个极短的向前向量
            fwd = seed_wp.transform.get_forward_vector()
            p0 = self.P[0] if len(self.P) > 0 else np.array(
                [seed_wp.transform.location.x, seed_wp.transform.location.y])
            p1 = p0 + np.array([fwd.x, fwd.y]) * step
            self.P = np.vstack([p0, p1])

        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])
        tang = np.diff(self.P, axis=0)
        tang = np.vstack([tang, tang[-1]])
        norm = np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        self.tang = tang / norm

        # 曲率计算
        if len(self.P) >= 3:
            psi = np.arctan2(self.tang[:, 1], self.tang[:, 0])
            dpsi = np.diff(psi)
            dpsi = (dpsi + np.pi) % (2 * np.pi) - np.pi
            ds_arr = np.diff(self.s) + 1e-9
            k_mid = dpsi / ds_arr
            self.kappa = np.zeros(len(self.s), dtype=float)
            self.kappa[1:-1] = 0.5 * (k_mid[:-1] + k_mid[1:])
            self.kappa[0], self.kappa[-1] = k_mid[0], k_mid[-1]
        else:
            self.kappa = np.zeros(len(self.s), dtype=float)

    def kappa_at_s(self, s: float) -> float:
        if len(self.s) < 2: return 0.0
        s = float(np.clip(s, self.s[0], self.s[-1]))
        return float(np.interp(s, self.s, self.kappa))

    def xy2se(self, x: float, y: float, max_proj_dist: Optional[float] = None) -> Tuple[
        Optional[float], Optional[float]]:
        if len(self.P) < 2: return None, None
        xy = np.array([x, y], dtype=float)

        v = xy - self.P[:-1]
        seg = self.P[1:] - self.P[:-1]
        seg_len2 = np.sum(seg ** 2, axis=1) + 1e-9
        t = np.clip(np.sum(v * seg, axis=1) / seg_len2, 0.0, 1.0)
        proj = self.P[:-1] + seg * t[:, None]
        dist2 = np.sum((proj - xy) ** 2, axis=1)

        i = int(np.argmin(dist2))
        min_dist = math.sqrt(dist2[i])

        if max_proj_dist is not None and min_dist > max_proj_dist:
            return None, None

        s_val = self.s[i] + t[i] * (self.s[i + 1] - self.s[i])
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        ey = (x - proj[i][0]) * nx + (y - proj[i][1]) * ny
        return float(s_val), float(ey)

    def se2xy(self, s: float, ey: float) -> Tuple[float, float]:
        if len(self.P) < 2: return self.P[0, 0], self.P[0, 1]
        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s) - 2))

        ds = max(1e-9, self.s[i + 1] - self.s[i])
        r = (s - self.s[i]) / ds
        base = self.P[i] * (1 - r) + self.P[i + 1] * r

        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        return float(base[0] + ey * nx), float(base[1] + ey * ny)


# ============================================================
# 2) 辅助函数
# ============================================================
def spawn_ego_from_scenario(world, scenario, env=None):
    print("\n--- [EGO SPAWN] ---")
    amap = world.get_map()
    ego_bp = get_ego_blueprint(world)
    tf = scenario.get_spawn_transform()
    if not tf: raise RuntimeError("Spawn point missing")

    tf.location.z += 0.2
    ego = world.try_spawn_actor(ego_bp, tf)
    if not ego:
        tf.location.z += 0.5
        ego = world.try_spawn_actor(ego_bp, tf)
    if not ego: raise RuntimeError("Spawn failed")

    if env: env.set_ego(ego)
    ego.set_simulate_physics(True)
    world.tick()
    return ego, amap.get_waypoint(ego.get_location(), project_to_road=True)


def spectator_follow_ego(world, ego, h=10.0, d=8.0, p=-30.0):
    if not ego: return
    t = ego.get_transform()
    loc = t.location - t.get_forward_vector() * d
    loc.z += h
    rot = carla.Rotation(pitch=p, yaw=t.rotation.yaw, roll=0)
    world.get_spectator().set_transform(carla.Transform(loc, rot))


def _lane_bound_points(wp):
    c = wp.transform.location
    r = wp.transform.get_right_vector()
    w = wp.lane_width
    return (carla.Location(c.x - r.x * w * 0.5, c.y - r.y * w * 0.5, c.z),
            carla.Location(c.x + r.x * w * 0.5, c.y + r.y * w * 0.5, c.z))


# ============================================================
# 5) Corridor
# ============================================================
def build_corridor(world, ego, ref, s_ahead=35.0, ds=1.0, expand_adjacent=False):
    amap = world.get_map()
    loc = ego.get_location()
    s0, _ = ref.xy2se(loc.x, loc.y)
    if s0 is None: return None

    s_nodes = np.arange(s0, s0 + s_ahead, ds)
    left_b, right_b = [], []

    L_MAR = 0.25

    for s in s_nodes:
        cx, cy = ref.se2xy(s, 0.0)
        wp = amap.get_waypoint(carla.Location(cx, cy, 0), project_to_road=True)

        ly, ry = 1.75, -1.75

        if wp:
            l_edge, r_edge = _lane_bound_points(wp)
            use_l, use_r = l_edge, r_edge

            if expand_adjacent:
                wl = wp.get_left_lane()
                wr = wp.get_right_lane()
                if wl and wl.lane_type == carla.LaneType.Driving:
                    use_l = _lane_bound_points(wl)[0]
                if wr and wr.lane_type == carla.LaneType.Driving:
                    use_r = _lane_bound_points(wr)[1]

            rvec = wp.transform.get_right_vector()
            l_final = carla.Location(use_l.x + rvec.x * L_MAR, use_l.y + rvec.y * L_MAR, use_l.z)
            r_final = carla.Location(use_r.x - rvec.x * L_MAR, use_r.y - rvec.y * L_MAR, use_r.z)

            _, ly = ref.xy2se(l_final.x, l_final.y)
            _, ry = ref.xy2se(r_final.x, r_final.y)

        left_b.append(float(ly))
        right_b.append(float(ry))

    left_b = np.array(left_b)
    right_b = np.array(right_b)

    # 2. 锥桶处理
    cones = []
    for a in world.get_actors():
        if "static.prop.constructioncone" not in a.type_id and "trafficcone" not in a.type_id: continue
        cloc = a.get_location()
        cs, cey = ref.xy2se(cloc.x, cloc.y)
        if cs and 0 < cs - s0 < s_ahead + 20:
            cones.append((cs, cey))

    bu_safe = left_b.copy()
    bl_safe = right_b.copy()

    ego_w = ego.bounding_box.extent.y * 2.0
    CONE_R = 0.4 + ego_w / 2 + 0.3

    for (cs, cey) in cones:
        mask = (s_nodes > cs - 10.0) & (s_nodes < cs + 6.0)
        if cey >= 0:
            bu_safe[mask] = np.minimum(bu_safe[mask], cey - CONE_R)
        else:
            bl_safe[mask] = np.maximum(bl_safe[mask], cey + CONE_R)

    # 3. 车辆避让
    cars = []
    for a in world.get_actors():
        if a.id == ego.id or "vehicle" not in a.type_id: continue
        cloc = a.get_location()
        cs, cey = ref.xy2se(cloc.x, cloc.y)
        if cs and 0 < cs - s0 < s_ahead + 20 and abs(cey) < 6.0:
            cars.append((cs, cey, a))

    CAR_R = ego_w / 2 + 0.5
    for (cs, cey, a) in cars:
        mask = (s_nodes > cs - 6.0) & (s_nodes < cs + 6.0)
        if cey >= 0:
            bu_safe[mask] = np.minimum(bu_safe[mask], cey - CAR_R)
        else:
            bl_safe[mask] = np.maximum(bl_safe[mask], cey + CAR_R)

    collision_mask = bu_safe <= bl_safe
    if np.any(collision_mask):
        mid = 0.5 * (bu_safe + bl_safe)
        bu_safe[collision_mask] = mid[collision_mask] + 0.1
        bl_safe[collision_mask] = mid[collision_mask] - 0.1

    return {
        "s": s_nodes, "bu": left_b, "bl": right_b,
        "bu_safe": bu_safe, "bl_safe": bl_safe,
        "mode": "NONE",
        "expanded": expand_adjacent
    }


# ============================================================
# 6) DP 规划器 (强制跟随 0.0)
# ============================================================
def plan_dp(s, bu, bl, preferred_ey=None):
    Ns = len(s)
    ey_grid = np.linspace(-6.0, 6.0, 61)
    Ny = len(ey_grid)
    INF = 1e9
    cost = np.full((Ns, Ny), INF)

    W_CEN = 1.0
    W_REF = 10.0
    W_PREF = 20.0
    W_SMOOTH = 15.0

    target_ey = np.zeros(Ns)
    if preferred_ey is not None:
        mask = ~np.isnan(preferred_ey)
        target_ey[mask] = preferred_ey[mask]
        w_map = np.full(Ns, W_REF)
        w_map[mask] = W_PREF
    else:
        w_map = np.full(Ns, W_REF)

    for i in range(Ns):
        l, u = bl[i], bu[i]
        valid = (ey_grid >= l) & (ey_grid <= u)

        mid = 0.5 * (l + u)
        c_node = w_map[i] * (ey_grid - target_ey[i]) ** 2 + W_CEN * (ey_grid - mid) ** 2

        cost[i, valid] = c_node[valid]
        if i == 0:
            # 找到离当前车道中心最近的可行点
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > 0:
                cost[0, valid_indices] = 0.0

    dp = np.full((Ns, Ny), INF)
    parent = np.zeros((Ns, Ny), dtype=int)
    dp[0] = cost[0]

    for i in range(1, Ns):
        for j in range(Ny):
            if cost[i, j] >= INF: continue
            j_min = max(0, j - 2)
            j_max = min(Ny, j + 3)
            prev_costs = dp[i - 1, j_min:j_max] + W_SMOOTH * (ey_grid[j] - ey_grid[j_min:j_max]) ** 2
            best_idx = np.argmin(prev_costs)
            dp[i, j] = cost[i, j] + prev_costs[best_idx]
            parent[i, j] = j_min + best_idx

    path_idx = np.zeros(Ns, dtype=int)
    path_idx[-1] = np.argmin(dp[-1])

    if dp[-1, path_idx[-1]] >= INF:
        path_idx[-1] = np.argmin(np.abs(ey_grid - target_ey[-1]))

    for i in range(Ns - 2, -1, -1):
        path_idx[i] = parent[i + 1, path_idx[i + 1]]

    return ey_grid[path_idx]


# ============================================================
# 7) RuleBasedPlanner (总控)
# ============================================================
class RuleBasedPlanner:
    def __init__(self, amap, v_ref_base=12.0):
        self.amap = amap
        self.v_ref_base = v_ref_base
        self.ref = None

        self.lc_state = "IDLE"
        self.target_lane_id = None
        self.lc_timer = 0.0
        self.enable_auto_lc = True
        self._last_draw = 0
        self._steer_prev = 0.0

    def update(self, world, ego, dt):
        # 1. 稳健地更新参考线
        wp = self.amap.get_waypoint(ego.get_location(), project_to_road=True)
        if not self.ref:
            self.ref = LaneRef(self.amap, wp)
        else:
            # 只有当偏离太远时才彻底重置，否则增量更新或重建
            self.ref = LaneRef(self.amap, wp)

        # 2. 构建走廊
        corridor = build_corridor(world, ego, self.ref, expand_adjacent=False)
        if not corridor: return 0, 0, 1, {}  # Fail safe

        # 3. 决策：路窄则扩宽
        s0, _ = self.ref.xy2se(ego.get_location().x, ego.get_location().y)
        min_w = 100.0
        if s0 is not None:
            mask = (corridor["s"] > s0) & (corridor["s"] < s0 + 20.0)
            if np.any(mask):
                min_w = np.min(corridor["bu_safe"][mask] - corridor["bl_safe"][mask])

        req_w = ego.bounding_box.extent.y * 2.2

        need_expand = False
        if self.lc_state == "COMMIT":
            need_expand = True
        elif self.enable_auto_lc and min_w < req_w:
            need_expand = True

        if need_expand:
            corridor = build_corridor(world, ego, self.ref, expand_adjacent=True)

            if self.lc_state == "IDLE":
                # 简单变道策略：哪边是 Driving Lane 就去哪边
                target = None
                if wp.get_left_lane() and wp.get_left_lane().lane_type == carla.LaneType.Driving:
                    target = wp.get_left_lane().lane_id
                elif wp.get_right_lane() and wp.get_right_lane().lane_type == carla.LaneType.Driving:
                    target = wp.get_right_lane().lane_id

                if target:
                    self.lc_state = "COMMIT"
                    self.target_lane_id = target
                    self.lc_timer = world.get_snapshot().timestamp.elapsed_seconds

        # 4. DP 目标 (NaN 默认走 0.0)
        pref_ey = np.full(len(corridor["s"]), np.nan)

        if self.lc_state == "COMMIT" and self.target_lane_id:
            for i, s in enumerate(corridor["s"]):
                cx, cy = self.ref.se2xy(s, 0.0)
                wp_curr = self.amap.get_waypoint(carla.Location(cx, cy, 0))

                tgt_wp = None
                if wp_curr.lane_id == self.target_lane_id:
                    tgt_wp = wp_curr
                elif wp_curr.get_left_lane() and wp_curr.get_left_lane().lane_id == self.target_lane_id:
                    tgt_wp = wp_curr.get_left_lane()
                elif wp_curr.get_right_lane() and wp_curr.get_right_lane().lane_id == self.target_lane_id:
                    tgt_wp = wp_curr.get_right_lane()

                if tgt_wp:
                    _, tey = self.ref.xy2se(tgt_wp.transform.location.x, tgt_wp.transform.location.y)
                    pref_ey[i] = tey

            if wp.lane_id == self.target_lane_id:
                self.lc_state = "IDLE"
                self.target_lane_id = None

        # 5. DP 规划 & 控制
        path_ey = plan_dp(corridor["s"], corridor["bu_safe"], corridor["bl_safe"], preferred_ey=pref_ey)
        return self.run_pure_pursuit(ego, corridor["s"], path_ey, corridor["bu_safe"], corridor["bl_safe"], dt)

    def run_pure_pursuit(self, ego, s_path, ey_path, bu, bl, dt):
        tf = ego.get_transform()
        vel = ego.get_velocity()
        v = math.hypot(vel.x, vel.y)

        s0, ey0 = self.ref.xy2se(tf.location.x, tf.location.y)
        if s0 is None: return 0, 0, 1, {}

        # 预瞄
        lookahead = max(4.0, v * 0.8)
        t_s = s0 + lookahead
        t_ey = np.interp(t_s, s_path, ey_path)

        # 计算转角
        tx, ty = self.ref.se2xy(t_s, t_ey)
        ego_yaw = math.radians(tf.rotation.yaw)
        dx = tx - tf.location.x
        dy = ty - tf.location.y
        local_x = dx * math.cos(-ego_yaw) - dy * math.sin(-ego_yaw)
        local_y = dx * math.sin(-ego_yaw) + dy * math.cos(-ego_yaw)

        k = 2.0 * local_y / (lookahead ** 2 + 1e-5)
        steer = k * 2.7
        steer = np.clip(steer, -1.0, 1.0)

        # 速度
        width_ahead = 10.0
        mask = (s_path > s0) & (s_path < s0 + 20.0)
        if np.any(mask):
            width_ahead = np.min(bu[mask] - bl[mask])

        v_target = self.v_ref_base
        if width_ahead < 2.2:
            v_target = 0.0
        elif width_ahead < 3.0:
            v_target = 5.0

        # 死锁救援：如果目标速度>1但车不动，给强力油门
        if v < 0.2 and v_target > 1.0:
            thr, brk = 0.8, 0.0
        else:
            err_v = v_target - v
            if err_v > 0:
                thr, brk = np.clip(err_v * 0.5, 0, 1), 0
            else:
                thr, brk = 0, np.clip(-err_v * 0.2, 0, 1)

        # Draw
        z = tf.location.z + 0.5
        if self.ref and len(self.ref.P) > 1:
            for i in range(len(ey_path) - 1):
                p1 = self.ref.se2xy(s_path[i], ey_path[i])
                p2 = self.ref.se2xy(s_path[i + 1], ey_path[i + 1])
                # world.debug.draw_line(carla.Location(p1[0], p1[1], z), carla.Location(p2[0], p2[1], z), 0.1, COL_DP, 0.1)

        return thr, steer, brk, {"v": v, "target": v_target}


# ============================================================
def main(scenario_type: str = "cones"):
    client = carla.Client("127.0.0.1", 2000);
    client.set_timeout(10.0)
    world = client.get_world()
    amap = world.get_map()
    TrafficFlowSpawner(client, world, 8000)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    planner = RuleBasedPlanner(amap)
    planner.enable_auto_lc = (scenario_type != "cones")

    if scenario_type == "cones":
        cfg = SimpleNamespace(cone_num=8, cone_step_behind=3.0, cone_step_lateral=0.4, cone_z_offset=0.5,
                              cone_lane_margin=0.25, cone_min_gap_from_junction=15.0, cone_grid=5.0,
                              spawn_min_gap_from_cone=25.0, tm_port=8000, enable_traffic_flow=True)
        scn = ConesScenario(world, amap, cfg)
    elif scenario_type == "trimma":
        cfg = SimpleNamespace(front_vehicle_distance=18.0, side_vehicle_offset=3.0, min_lane_count=3, tm_port=8000,
                              tm_global_distance=2.5, front_speed_diff_pct=85.0, side_speed_diff_pct=80.0,
                              disable_lane_change=True, enable_traffic_flow=True)
        scn = TrimmaScenario(world, amap, cfg)
        planner.enable_overtake = True
    elif scenario_type == "construction":
        cfg = SimpleNamespace(construction_distance=30.0, construction_length=20.0, traffic_density=3.0,
                              traffic_speed=8.0, min_gap_for_lane_change=12.0, construction_type="construction1",
                              flow_range=80.0, tm_port=8000, enable_traffic_flow=True)
        scn = ConstructionLaneChangeScenario(world, amap, cfg)
    elif scenario_type == "jaywalker":
        cfg = SimpleNamespace(jaywalker_distance=25.0, jaywalker_speed=2.5, jaywalker_trigger_distance=18.0,
                              jaywalker_start_side="random", use_occlusion_vehicle=False, tm_port=8000,
                              enable_traffic_flow=True)
        scn = JaywalkerScenario(world, amap, cfg)
    else:
        return

    scn.setup()
    ego, _ = spawn_ego_from_scenario(world, scn)

    eva = None
    if EvaMonitor:
        try:
            eva = EvaMonitor(); eva.attach(world, ego)
        except:
            pass

    try:
        while True:
            spectator_follow_ego(world, ego)
            t, s, b, dbg = planner.update(world, ego, 0.05)
            ego.apply_control(carla.VehicleControl(throttle=t, steer=s, brake=b))
            world.tick()
            if scenario_type == "jaywalker": scn.check_and_trigger(ego.get_location()); scn.tick_update()

            if eva:
                try:
                    eva.render(eva.tick())
                    import pygame;
                    pygame.event.pump()
                except:
                    pass
    except KeyboardInterrupt:
        pass
    finally:
        scn.cleanup();
        ego.destroy()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="cones")
    args = parser.parse_args()
    main(args.scenario)