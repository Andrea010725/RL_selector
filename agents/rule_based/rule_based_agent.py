# agents/rule_based/agent.py
from __future__ import annotations
import math
import random
import sys
sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np

from typing import List, Tuple
sys.path.append("/home/ajifang/czw/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.highway_obs import right_unit_vector_from_yaw, forward_unit_vector_from_yaw
from env.highway_obs import shift_location
from planning.dp_corridor import DPCorridor
from planning.obstacles import collect_obstacles_api
from agents.rule_based.vis_debug import draw_corridor, draw_ego_marker, TelemetryLogger, draw_pts_se
from env.highway_obs import set_spectator_follow_actor, set_spectator_fixed


# ========= 1) 在“同一车道中心 & 上游15–20m”生成 EGO =========
def spawn_ego_upstream_lane_center(env: HighwayEnv) -> carla.Actor:
    world = env.world
    first_tf = env.get_first_cone_transform()
    if world is None or first_tf is None:
        raise RuntimeError("环境未就绪，缺少第一个锥桶位姿。")
    amap: carla.Map = world.get_map()

    yaw_cone = first_tf.rotation.yaw
    def ang_diff_deg(a, b):
        d = a - b
        while d > 180.0: d -= 360.0
        while d < -180.0: d += 360.0
        return abs(d)

    base_wp = amap.get_waypoint(first_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if base_wp is None:
        raise RuntimeError("无法将锥桶位置投影到驾驶车道。")

    candidates = [base_wp]
    wp = base_wp
    for _ in range(2):
        left = wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            candidates.append(left); wp = left
        else:
            break
    wp = base_wp
    for _ in range(2):
        right = wp.get_right_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            candidates.append(right); wp = right
        else:
            break

    best = min(candidates, key=lambda w: ang_diff_deg(w.transform.rotation.yaw, yaw_cone))
    lane_wp = best
    target_ids = (lane_wp.road_id, lane_wp.section_id, lane_wp.lane_id)

    offset = random.uniform(15.0, 20.0)
    cand_list = lane_wp.previous(offset)
    if not cand_list:
        step_back, acc, wp = 2.0, 0.0, lane_wp
        while acc < 30.0:
            prevs = wp.previous(step_back)
            if not prevs: break
            wp = prevs[0]; acc += step_back
            if (wp.road_id, wp.section_id, wp.lane_id) == target_ids:
                cand_list = [wp]; break
    if not cand_list:
        raise RuntimeError("沿车道上游回退失败，未找到EGO生成点。")

    spawn_wp = cand_list[0]
    if (spawn_wp.road_id, spawn_wp.section_id, spawn_wp.lane_id) != target_ids:
        step_back, trials, wp = 1.0, 0, spawn_wp
        while trials < 20:
            prevs = wp.previous(step_back)
            if not prevs: break
            wp = prevs[0]; trials += 1
            if (wp.road_id, wp.section_id, wp.lane_id) == target_ids:
                spawn_wp = wp; break

    yaw = float(spawn_wp.transform.rotation.yaw)
    spawn_tf = carla.Transform(
        location=carla.Location(
            x=spawn_wp.transform.location.x,
            y=spawn_wp.transform.location.y,
            z=spawn_wp.transform.location.z + 0.20
        ),
        rotation=carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    )

    ego_bp = get_ego_blueprint(world)
    jitters = [(0,0), (0.4,0), (-0.4,0), (0,0.4), (0,-0.4)]
    for dx, dy in jitters:
        rad = math.radians(yaw)
        rx, ry = math.sin(rad), -math.cos(rad)
        fx, fy = math.cos(rad),  math.sin(rad)
        loc = carla.Location(
            x=spawn_tf.location.x + rx*dx + fx*dy,
            y=spawn_tf.location.y + ry*dx + fy*dy,
            z=spawn_tf.location.z
        )
        try_tf = carla.Transform(loc, spawn_tf.rotation)
        ego = world.try_spawn_actor(ego_bp, try_tf)
        if ego is not None:
            print(f"[EGO] 生成 @ lane_center road/section/lane=({spawn_wp.road_id}/{spawn_wp.section_id}/{spawn_wp.lane_id})")
            env.set_ego(ego)
            return ego
    raise RuntimeError("未能在目标车道中心附近生成 EGO。")


# ========= 2) 同车道参考线 =========
class LaneRef:
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len: float = 500.0):
        pts, wps = [], []
        self.wps = wps
        wp = seed_wp
        dist = 0.0
        guard_ids = (wp.road_id, wp.section_id, wp.lane_id)
        pts.append((wp.transform.location.x, wp.transform.location.y)); wps.append(wp)
        while dist < max_len:
            nxts = wp.next(step)
            if not nxts: break
            wp = nxts[0]
            if (wp.road_id, wp.section_id, wp.lane_id) != guard_ids: break
            pts.append((wp.transform.location.x, wp.transform.location.y)); wps.append(wp)
            dist += step
        self.P = np.asarray(pts, dtype=float)
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])
        tang = np.diff(self.P, axis=0)
        tang = np.vstack([tang, tang[-1]])
        self.tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        self.guard_ids = guard_ids
        self.step = step
        self.wps = wps             # FIX: 暴露 wps，给 map_lane_walls_with_ref 使用  ✅

    def _segment_index_and_t(self, x, y):
        P = self.P; xy = np.array([x,y], dtype=float)
        seg = P[1:] - P[:-1]
        v = xy - P[:-1]
        seg_len2 = (seg[:,0]**2 + seg[:,1]**2) + 1e-9
        t = np.clip((v[:,0]*seg[:,0] + v[:,1]*seg[:,1]) / seg_len2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:,None]
        dist2 = np.sum((proj - xy[None,:])**2, axis=1)
        i = int(np.argmin(dist2))
        return i, float(t[i]), proj[i]

    def xy2se(self, x: float, y: float):
        i, t, proj = self._segment_index_and_t(x, y)
        s_val = self.s[i] + t * (self.s[i+1] - self.s[i])
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        ey = float((x - proj[0]) * nx + (y - proj[1]) * ny)
        return float(s_val), float(ey)

    def se2xy(self, s: float, ey: float):
        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s)-2))
        ratio = (s - self.s[i]) / max(1e-9, self.s[i+1]-self.s[i])
        base = self.P[i] * (1 - ratio) + self.P[i+1] * ratio
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        x = base[0] + ey * nx
        y = base[1] + ey * ny
        return float(x), float(y)


# ========= 3) 规则型 Planner =========
class RuleBasedPlanner:
    def __init__(self, dp: DPCorridor, ref: LaneRef, v_ref_base=12.0, dp_interval=8):
        self.dp = dp
        self.ref = ref
        self.v_ref_base = v_ref_base
        self.dp_interval = dp_interval
        self.corridor = None
        self._prev_delta = 0.0
        self._prev_ax = 0.0

    @staticmethod
    def _finite(x):
        return np.isfinite(x).all()

    def _fallback_lane_box(self):
        # 没有走廊时的保底：±1.75m
        return -1.75, +1.75

    def update_corridor(self, world, ego=None, debug_draw_points: bool = True):
        if ego is None:
            self.corridor = None
            return

        ego_tf = ego.get_transform()
        s_center, _ = self.ref.xy2se(ego_tf.location.x, ego_tf.location.y)
        self.dp.set_window(s0=s_center - 10.0, length_m=30.0)

        # 2) 车道线（始终存在）
        map_left, map_right = self.dp.map_lane_walls_with_ref(world, self.ref, safety=0.25)

        # 3) 障碍硬墙
        pts_se = collect_obstacles_api(world, ego, self.ref.xy2se,
                                       s_center=s_center, s_back=10.0, s_fwd=20.0,
                                       r_xy=35.0, horizon_T=2.0, dt=0.2, static_density=0.20)
        hard_left, hard_right = self.dp.hard_walls_from_points(
            pts_se, self.dp.s_grid, self.dp.ds, s_dilate=1.0, safety=0.20
        )
        hard_left = np.minimum(hard_left, map_left)
        hard_right = np.maximum(hard_right, map_right)

        # 4) 融合 + 保底
        corr = self.dp.fuse_corridor(
            map_left=map_left, map_right=map_right,
            hard_left=hard_left, hard_right=hard_right,
            min_width=1.8, temporal_alpha=0.6
        )

        # NEW: 若仍有 NaN/异常宽度，立即保底为默认车道盒
        lo_bad = (~self._finite(corr.lower)) | (~self._finite(corr.upper)) | ((corr.upper - corr.lower) < 0.6)
        if np.any(lo_bad):
            lo0, up0 = self._fallback_lane_box()
            corr.lower[lo_bad] = lo0
            corr.upper[lo_bad] = up0

        self.corridor = corr

        if debug_draw_points and pts_se:
            draw_pts_se(world, self.ref, pts_se, color=(0, 255, 0), size=0.08, life=0.8)

    def compute_control(self, obs: dict, dt: float = 0.05) -> tuple[float, float, float, dict]:
        ego_pose = obs.get("ego_pose", {})
        ego_v = obs.get("ego_v", {})
        x, y = float(ego_pose.get("x", 0.0)), float(ego_pose.get("y", 0.0))
        yaw_deg = float(ego_pose.get("yaw", 0.0))
        speed = float(ego_v.get("speed", 0.0))
        yaw_rad = np.deg2rad(yaw_deg)

        s_now, ey_now = self.ref.xy2se(x, y)

        if self.corridor is None:
            lo, up = self._fallback_lane_box()   # 启动保底
            s_q = s_now
        else:
            s_min, s_max = float(self.corridor.s[0]), float(self.corridor.s[-1])
            s_q = float(np.clip(s_now, s_min, s_max))
            lo = float(np.interp(s_q, self.corridor.s, self.corridor.lower))
            up = float(np.interp(s_q, self.corridor.s, self.corridor.upper))

        if not np.isfinite(lo) or not np.isfinite(up) or (up - lo) < 0.6:
            lo, up = self._fallback_lane_box()   # 进一步保底

        if lo > up:
            lo, up = up, lo
        width = max(1e-3, up - lo)

        margin_r = ey_now - lo
        margin_l = up - ey_now

        # 低速时抑制航向项，避免原地打方向
        ds_yaw = max(2.0, speed * 0.3)
        x_fwd, y_fwd = self.ref.se2xy(s_q + ds_yaw, 0.0)
        dx, dy = x_fwd - x, y_fwd - y
        if math.hypot(dx, dy) < 0.1:
            e_psi = 0.0
        else:
            yaw_ref = np.arctan2(dy, dx)
            e_psi = np.arctan2(np.sin(yaw_ref - yaw_rad), np.cos(yaw_ref - yaw_rad))

        Ky0, Kpsi0 = 0.6, 1.0
        # VERY IMPORTANT: 低速抑制航向项
        heading_scale = 0.2 if speed < 0.5 else 1.0  # NEW
        Ky = Ky0 / (1.0 + 0.10 * max(0.0, speed))
        Kpsi = heading_scale * (Kpsi0 / (1.0 + 0.08 * max(0.0, speed)))

        bias_gain, bias_zone = 0.25, 0.40
        bias = 0.0
        if 0 <= margin_r < bias_zone and margin_r <= margin_l:
            bias = +bias_gain * (bias_zone - margin_r)
        elif 0 <= margin_l < bias_zone and margin_l < margin_r:
            bias = -bias_gain * (bias_zone - margin_l)

        ey_ref = 0.5 * (lo + up) + bias
        e_y = ey_ref - ey_now

        min_margin = min(abs(margin_l), abs(margin_r))
        guard = 1.0 + 0.6 * np.exp(-min_margin / 0.25)

        delta_cmd = guard * (Ky * e_y + Kpsi * e_psi)
        delta_cmd = float(np.clip(delta_cmd, -0.30, 0.30))
        max_d_delta = np.deg2rad(45.0) * dt
        d_delta = float(np.clip(delta_cmd - self._prev_delta, -max_d_delta, +max_d_delta))
        delta_cmd = self._prev_delta + d_delta
        self._prev_delta = delta_cmd

        max_front_wheel_angle = 0.35
        steer = float(np.clip(delta_cmd / max_front_wheel_angle, -1.0, 1.0))

        v_base = float(self.v_ref_base)
        v_ref_w = np.clip(v_base * (width / 3.0), 6.0, v_base)
        near = min(margin_l, margin_r)
        shrink = 1.0 if near >= 0.6 else max(0.4, near / 0.6)
        v_ref = v_ref_w * shrink

        ax_cmd = 0.8 * (v_ref - speed)
        ax_cmd = float(np.clip(ax_cmd, -3.0, 2.0))
        max_d_ax = 3.0 * dt
        d_ax = float(np.clip(ax_cmd - self._prev_ax, -max_d_ax, +max_d_ax))
        ax = self._prev_ax + d_ax
        self._prev_ax = ax

        if ax >= 0.0:
            throttle, brake = ax / 2.0, 0.0
        else:
            throttle, brake = 0.0, (-ax) / 3.0
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        dbg = dict(
            mode="LaneWalls ∩ HardWalls  +  RuleMidline",
            s=s_now, s_q=s_q, ey=ey_now, lo=lo, up=up, width=width,
            ey_ref=ey_ref, e_y=e_y, e_psi=e_psi, guard=guard,
            v=speed, v_ref=v_ref, ax=ax, delta=delta_cmd,
            throttle=throttle, brake=brake, steer=steer,
            margin_l=margin_l, margin_r=margin_r, bias=bias
        )
        return throttle, steer, brake, dbg

    def hard_world_cleanup(world: carla.World):
        try:
            actors = world.get_actors()
            for v in actors.filter("vehicle.*"):
                try:
                    if v.attributes.get("role_name", "") == "hero":
                        v.destroy()
                except Exception:
                    pass
            for c in actors:
                try:
                    if ("trafficcone" in c.type_id) or ("static.prop.cone" in c.type_id):
                        c.destroy()
                except Exception:
                    pass
        except Exception:
            pass


# ========= 4) 主流程 =======
def main():
    env = HighwayEnv(host="127.0.0.1", port=2000, sync=True, fixed_dt=0.05).connect()
    logger = None
    try:
        env.setup_scene(
            num_cones=10, step_forward=3.0, step_right=0.30,
            z_offset=0.0, min_gap_from_junction=15.0, grid=5.0, set_spectator=True,
        )
        ego = spawn_ego_upstream_lane_center(env)

        set_spectator_fixed(world=env.world, ego=ego, back=5.0,height=7.0,side_offset=0, # 右偏一点，能看到车身
                            pitch_deg=0.0, look_at_roof=True )


        first_tf = env.get_first_cone_transform()
        amap = env.world.get_map()
        seed_wp = amap.get_waypoint(first_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ref = LaneRef(amap, seed_wp=seed_wp, step=1.0, max_len=500.0)

        dp = DPCorridor(s_max=120.0, ds=2.0, ey_span=3.0, dey=0.2,
                        obs_sigma=0.6, smooth_w=0.05, max_step=2)
        planner = RuleBasedPlanner(dp, ref, v_ref_base=12.0, dp_interval=8)

        logger = TelemetryLogger(out_dir="logs_rule_based")
        dt = 0.05
        frame = 0

        # 先更新一次走廊（重要）
        planner.update_corridor(env.world, ego=env.ego)
        draw_corridor(env.world, ref, planner.corridor)

        while True:
            if frame % planner.dp_interval == 0:
                planner.update_corridor(env.world, ego=env.ego)
                draw_corridor(env.world, ref, planner.corridor)

            obs, _ = env.step()
            throttle, steer, brake, dbg = planner.compute_control(obs, dt=dt)

            if frame % 2 == 0:
                ego_pose = obs.get("ego_pose", {})
                draw_ego_marker(env.world, ego_pose.get("x", 0.0), ego_pose.get("y", 0.0))

            if frame % 10 == 0:
                print(
                    f"[CTRL] {dbg['mode']} | s={dbg['s']:.1f}, ey={dbg['ey']:.2f}, "
                    f"lo={dbg['lo']:.2f}, up={dbg['up']:.2f}, width={dbg['width']:.2f} "
                    f"| ey_ref={dbg['ey_ref']:.2f}, e_y={dbg['e_y']:.2f} "
                    f"| v={dbg['v']:.2f}->{dbg['v_ref']:.2f} ax={dbg['ax']:.2f} "
                    f"| δ={dbg['delta']:.3f} -> steer={dbg['steer']:.2f} "
                    f"| throttle={dbg['throttle']:.2f}, brake={dbg['brake']:.2f}"
                )

            set_spectator_follow_actor(world=env.world, actor=env.ego, mode="chase", distance=28.0, height=7.0,
                                       pitch_deg=-12.0, yaw_offset=0.0, side_offset=2.0)

            env.apply_control(throttle=throttle, steer=steer, brake=brake)
            logger.log(frame, obs, dbg, ref)
            frame += 1

    except KeyboardInterrupt:
        print("\n[Stop] 手动退出。")
    finally:
        try:
            if logger is not None:
                logger.save_csv(); logger.plot()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            planner.hard_world_cleanup(env.world)
        except Exception:
            pass


if __name__ == "__main__":
    main()
