# agents/rule_based/agent.py
from __future__ import annotations
import math
import sys
import numpy as np
from types import SimpleNamespace
from typing import Optional, Tuple

sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla

# 工程依赖
sys.path.append("/home/ajifang/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.scenarios import ConesScenario, JaywalkerScenario, TrimmaScenario, ConstructionLaneChangeScenario


# ============================================================
# 1) LaneRef：稳健的参考线
# ============================================================
class LaneRef:
    def __init__(self, amap, seed_wp, step=1.0, max_len=150.0):
        pts = []
        wp = seed_wp
        dist = 0.0
        pts.append((wp.transform.location.x, wp.transform.location.y))

        while dist < max_len:
            nxts = wp.next(step)
            if not nxts: break
            best_wp, max_dot = None, -1.0
            fwd = wp.transform.get_forward_vector()
            for n_cand in nxts:
                if n_cand.lane_type != carla.LaneType.Driving: continue
                vec = n_cand.transform.location - wp.transform.location
                norm = math.sqrt(vec.x ** 2 + vec.y ** 2) + 1e-9
                dot = (vec.x * fwd.x + vec.y * fwd.y) / norm
                if dot > max_dot:
                    max_dot, best_wp = dot, n_cand
            if best_wp is None or (dist > 5.0 and max_dot < 0.5): break
            wp = best_wp
            pts.append((wp.transform.location.x, wp.transform.location.y))
            dist += step

        self.P = np.asarray(pts)
        if len(self.P) < 2:
            fwd = seed_wp.transform.get_forward_vector()
            p0 = np.array([seed_wp.transform.location.x, seed_wp.transform.location.y])
            self.P = np.vstack([p0, p0 + np.array([fwd.x, fwd.y]) * step])

        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])
        tang = np.diff(self.P, axis=0)
        self.tang = np.vstack([tang, tang[-1]]) / (np.linalg.norm(tang[-1]) + 1e-9)

    def xy2se(self, x: float, y: float):
        xy = np.array([x, y])
        v = xy - self.P[:-1]
        seg = self.P[1:] - self.P[:-1]
        t = np.clip(np.sum(v * seg, axis=1) / (np.sum(seg ** 2, axis=1) + 1e-9), 0.0, 1.0)
        proj = self.P[:-1] + seg * t[:, None]
        dist2 = np.sum((proj - xy) ** 2, axis=1)
        i = np.argmin(dist2)
        s_val = self.s[i] + t[i] * (self.s[i + 1] - self.s[i])
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        ey = (x - proj[i][0]) * nx + (y - proj[i][1]) * ny
        return float(s_val), float(ey)

    def se2xy(self, s, ey):
        s = np.clip(s, self.s[0], self.s[-1])
        i = np.searchsorted(self.s, s) - 1
        i = max(0, min(i, len(self.s) - 2))
        r = (s - self.s[i]) / (self.s[i + 1] - self.s[i] + 1e-9)
        base = self.P[i] * (1 - r) + self.P[i + 1] * r
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        return float(base[0] + ey * nx), float(base[1] + ey * ny)


# ============================================================
# 2) 核心组件：走廊与规划器
# ============================================================
def _lane_bound_points(wp):
    c, r, w = wp.transform.location, wp.transform.get_right_vector(), wp.lane_width
    return (carla.Location(c.x - r.x * w * 0.5, c.y - r.y * w * 0.5, c.z),
            carla.Location(c.x + r.x * w * 0.5, c.y + r.y * w * 0.5, c.z))


def build_corridor(world, ego, ref, s_ahead=40.0, ds=1.0, expand_adjacent=False):
    amap = world.get_map()
    loc = ego.get_location()
    s0, _ = ref.xy2se(loc.x, loc.y)
    if s0 is None: return None

    s_nodes = np.arange(s0, s0 + s_ahead, ds)
    bu_list, bl_list = [], []
    L_MAR = 0.1

    for s in s_nodes:
        cx, cy = ref.se2xy(s, 0.0)
        wp = amap.get_waypoint(carla.Location(cx, cy, 0), project_to_road=True)
        ly, ry = 1.75, -1.75
        if wp:
            use_l, use_r = _lane_bound_points(wp)
            if expand_adjacent:
                wl, wr = wp.get_left_lane(), wp.get_right_lane()
                if wl and wl.lane_type == carla.LaneType.Driving: use_l = _lane_bound_points(wl)[0]
                if wr and wr.lane_type == carla.LaneType.Driving: use_r = _lane_bound_points(wr)[1]
            rvec = wp.transform.get_right_vector()
            l_f = carla.Location(use_l.x + rvec.x * L_MAR, use_l.y + rvec.y * L_MAR, use_l.z)
            r_f = carla.Location(use_r.x - rvec.x * L_MAR, use_r.y - rvec.y * L_MAR, use_r.z)
            _, ey_l = ref.xy2se(l_f.x, l_f.y)
            _, ey_r = ref.xy2se(r_f.x, r_f.y)
            ly, ry = max(ey_l, ey_r), min(ey_l, ey_r)
        bu_list.append(ly)
        bl_list.append(ry)

    bu_safe, bl_safe = np.array(bu_list), np.array(bl_list)
    ego_w = ego.bounding_box.extent.y * 2.0
    CONE_R = (ego_w / 2.0) + 0.3  # 加强侧向排斥感
    WALKER_R = (ego_w / 2.0) + 1.0

    for a in world.get_actors():
        if a.id == ego.id: continue
        type_id = a.type_id.lower()
        if "walker" in type_id or "vehicle" in type_id or "cone" in type_id:
            cloc = a.get_location()
            cs, cey = ref.xy2se(cloc.x, cloc.y)
            if cs and 0 < cs - s0 < s_ahead + 5:
                is_walker = "walker" in type_id
                s_margin = 5.0 if is_walker else 4.0
                mask = (s_nodes > cs - s_margin) & (s_nodes < cs + s_margin)
                r_inflated = WALKER_R if is_walker else CONE_R
                mid_lane = (bu_safe + bl_safe) / 2.0
                idx = max(0, min(np.searchsorted(s_nodes, cs) - 1, len(mid_lane) - 1))
                if cey > mid_lane[idx]:
                    bu_safe[mask] = np.minimum(bu_safe[mask], cey - r_inflated)
                else:
                    bl_safe[mask] = np.maximum(bl_safe[mask], cey + r_inflated)
    return {"s": s_nodes, "bu_safe": bu_safe, "bl_safe": bl_safe}


def plan_dp(s, bu, bl):
    Ns, Ny = len(s), 61
    ey_grid = np.linspace(-7.0, 7.0, Ny)
    cost = np.full((Ns, Ny), 1e6)
    for i in range(Ns):
        valid = (ey_grid >= bl[i]) & (ey_grid <= bu[i])
        if not np.any(valid):
            cost[i, np.argmin(np.abs(ey_grid - (bu[i] + bl[i]) / 2))] = 10.0
        else:
            # 极大降低对中心线的依赖，鼓励偏离中心超车
            dist_to_mid = (ey_grid[valid] - (bu[i] + bl[i]) / 2) ** 2
            cost[i, valid] = 0.5 * (ey_grid[valid]) ** 2 + 2.0 * dist_to_mid

    dp = np.full((Ns, Ny), 1e9)
    parent = np.zeros((Ns, Ny), dtype=int)
    dp[0] = cost[0]
    for i in range(1, Ns):
        for j in range(Ny):
            if cost[i, j] >= 1e6: continue
            # 调低变道惩罚，让超车动作更果断
            prev_costs = dp[i - 1] + 3.0 * (ey_grid[j] - ey_grid) ** 2
            best = np.argmin(prev_costs)
            dp[i, j] = cost[i, j] + prev_costs[best]
            parent[i, j] = best

    path_idx = np.zeros(Ns, dtype=int)
    path_idx[-1] = np.argmin(dp[-1])
    for i in range(Ns - 2, -1, -1): path_idx[i] = parent[i + 1, path_idx[i + 1]]
    return ey_grid[path_idx]


# ============================================================
# 3) Planner 主类
# ============================================================
class RuleBasedPlanner:
    def __init__(self, amap, v_ref_base=10.0):
        self.amap = amap
        self.v_ref_base = v_ref_base
        self.ref = None

    def update_corridor(self, world, ego):
        loc = ego.get_location()
        wp = self.amap.get_waypoint(loc, project_to_road=True)
        self.ref = LaneRef(self.amap, wp)

        # 始终允许探测相邻车道，以便发现超车机会
        corridor = build_corridor(world, ego, self.ref, expand_adjacent=True)
        path_ey = plan_dp(corridor["s"], corridor["bu_safe"], corridor["bl_safe"])

        return self.run_control(ego, corridor["s"], path_ey, corridor["bu_safe"], corridor["bl_safe"], world)

    def run_control(self, ego, s_path, ey_path, bu, bl, world):
        tf = ego.get_transform()
        vel = ego.get_velocity()
        v = math.hypot(vel.x, vel.y)
        s0, _ = self.ref.xy2se(tf.location.x, tf.location.y)

        # 转向控制：前视距离随速度增加
        lh = max(4.0, v * 0.8)
        t_ey = np.interp(s0 + lh, s_path, ey_path)
        tx, ty = self.ref.se2xy(s0 + lh, t_ey)
        yaw = math.radians(tf.rotation.yaw)
        dx, dy = tx - tf.location.x, ty - tf.location.y
        ly = dx * math.sin(-yaw) + dy * math.cos(-yaw)
        steer = np.clip((2.0 * ly / (lh ** 2 + 1e-5)) * 3.0, -1.0, 1.0)

        # 纵向跟车限速逻辑
        v_acc = self.v_ref_base
        min_dist = 100.0
        for a in world.get_actors():
            if "vehicle" in a.type_id and a.id != ego.id:
                cloc = a.get_location()
                cs, cey = self.ref.xy2se(cloc.x, cloc.y)
                dist_s = cs - s0
                path_ey_at_cs = np.interp(cs, s_path, ey_path)

                # 如果前车在我的规划路径上，且距离过近
                if 0 < dist_s < 18.0 and abs(cey - path_ey_at_cs) < 1.4:
                    min_dist = min(min_dist, dist_s)
                    # 距离4米时完全停止，留出变道余量
                    v_acc = min(v_acc, max(0.0, (dist_s - 4.5) * 1.8))

        # 宽度限速
        mask = (s_path > s0) & (s_path < s0 + 10.0)
        w_ahead = np.min(np.abs(bu[mask] - bl[mask])) if np.any(mask) else 10.0
        ego_w = ego.bounding_box.extent.y * 2.0

        if w_ahead < 0.5:
            v_width = 0.0
        elif w_ahead < ego_w + 0.2:
            v_width = 2.0
        else:
            v_width = self.v_ref_base

        v_target = min(v_width, v_acc)

        # 动力控制
        err = v_target - v
        if v < 0.2 and v_target > 0.5:
            t, b = 0.8, 0.0
        else:
            t = np.clip(err * 0.5, 0, 0.8) if err > 0 else 0
            b = np.clip(-err * 0.6, 0, 1.0) if err < 0 else 0

        return t, steer, b, {"v": v, "w": w_ahead, "vt": v_target, "dist": min_dist}


# ============================================================
# Main
# ============================================================
def main():
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    amap = world.get_map()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    cfg = SimpleNamespace(cone_num=8, cone_step_behind=3.0, cone_step_lateral=0.4, cone_z_offset=0.5,
                          cone_lane_margin=0.25, cone_min_gap_from_junction=15.0, cone_grid=5.0,
                          spawn_min_gap_from_cone=25.0, tm_port=8000, enable_traffic_flow=False)

    # 根据需要切换场景：ConesScenario, JaywalkerScenario, TrimmaScenario
    # scn = TrimmaScenario(world, amap, cfg)
    scn = ConstructionLaneChangeScenario(world, amap, cfg)
    scn.setup()

    ego_bp = get_ego_blueprint(world)
    ego = world.spawn_actor(ego_bp, scn.get_spawn_transform())
    planner = RuleBasedPlanner(amap)

    print("\n--- 启动 Agent (已优化超车与ACC) ---")
    try:
        while True:
            t = ego.get_transform()
            loc = t.location - t.get_forward_vector() * 8.0 + carla.Location(z=5.0)
            world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-30, yaw=t.rotation.yaw)))

            if hasattr(scn, 'check_and_trigger'):
                scn.check_and_trigger(ego.get_location())
            if hasattr(scn, 'tick_update'):
                scn.tick_update()

            thr, steer, brk, dbg = planner.update_corridor(world, ego)
            ego.apply_control(carla.VehicleControl(throttle=thr, steer=steer, brake=brk))
            world.tick()

            print(f"VT: {dbg['vt']:.1f} | V: {dbg['v']:.1f} | W: {dbg['w']:.2f} | Dist: {dbg.get('dist', 0):.1f}   ",
                  end='\r')
    finally:
        ego.destroy()
        world.apply_settings(carla.WorldSettings(synchronous_mode=False))


if __name__ == "__main__":
    main()
