# agents/rule_based/agent.py
from __future__ import annotations
import math
import random

import sys
sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np

sys.path.append("/home/ajifang/czw/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.highway_obs import right_unit_vector_from_yaw, forward_unit_vector_from_yaw
from env.highway_obs import shift_location
from planning.dp_corridor import DPCorridor
from utils.control_mapping import ax_to_throttle_brake, delta_to_steer
from agents.rule_based.vis_debug import draw_corridor, draw_ego_marker, TelemetryLogger
from env.highway_obs import set_spectator_follow_actor

# ========= 1) 在“同一车道中心 & 上游15–20m”生成 EGO =========
def spawn_ego_upstream_lane_center(env: HighwayEnv) -> carla.Actor:
    """
    在第一个锥桶“前”（上游）15~20m，且与锥桶所在【同一驾驶车道、同向】的【车道中心】生成 EGO。
    关键修正：
      - 先按锥桶朝向 yaw 选择与其夹角最小的 Driving 车道（避免吸到对向车道）
      - 再在该 lane 上 previous(15~20m) 回退生成点
    """
    world = env.world
    first_tf = env.get_first_cone_transform()
    if world is None or first_tf is None:
        raise RuntimeError("环境未就绪，缺少第一个锥桶位姿。")
    amap: carla.Map = world.get_map()

    yaw_cone = first_tf.rotation.yaw  # 锥桶朝向（即道路行进方向）
    yaw_cone_rad = math.radians(yaw_cone)

    # —— 1) 选“与锥桶朝向匹配”的 Driving 车道中心 —— #
    base_wp = amap.get_waypoint(first_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if base_wp is None:
        raise RuntimeError("无法将锥桶位置投影到驾驶车道。")

    # 枚举邻接车道（中心、左、右各最多2层），挑与 yaw_cone 夹角最小的
    def yaw_of_wp(wp: carla.Waypoint) -> float:
        return float(wp.transform.rotation.yaw)

    def ang_diff_deg(a_deg: float, b_deg: float) -> float:
        d = a_deg - b_deg
        while d > 180.0: d -= 360.0
        while d < -180.0: d += 360.0
        return abs(d)

    candidates = [base_wp]
    # 向左找两条
    wp = base_wp
    for _ in range(2):
        left = wp.get_left_lane()
        if left and left.lane_type == carla.LaneType.Driving:
            candidates.append(left)
            wp = left
        else:
            break
    # 向右找两条
    wp = base_wp
    for _ in range(2):
        right = wp.get_right_lane()
        if right and right.lane_type == carla.LaneType.Driving:
            candidates.append(right)
            wp = right
        else:
            break

    # 选择与锥桶朝向夹角最小的那条 Driving 车道
    best = min(candidates, key=lambda w: ang_diff_deg(yaw_of_wp(w), yaw_cone))
    misalign = ang_diff_deg(yaw_of_wp(best), yaw_cone)
    if misalign > 45.0:
        # 如果所有候选都大于45°，很可能在路口/几何复杂处；此时仍用最近中心，但会打印告警
        print(f"[EGO-SPAWN][WARN] 所有候选与锥桶朝向夹角>45° (min={misalign:.1f}°)，将使用最近Driving车道中心。")
        best = base_wp

    lane_wp = best  # 经过朝向筛选的“同向”车道
    target_ids = (lane_wp.road_id, lane_wp.section_id, lane_wp.lane_id)

    # —— 2) 在“同向同车道”上游回退 15~20m —— #
    offset = random.uniform(15.0, 20.0)
    cand_list = lane_wp.previous(offset)
    if not cand_list:
        # 逐步兜底回退（最多30m）
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

    # 保证仍在同一条 lane（防止 previous 穿越到相邻 lane）
    if (spawn_wp.road_id, spawn_wp.section_id, spawn_wp.lane_id) != target_ids:
        step_back, trials, wp = 1.0, 0, spawn_wp
        while trials < 20:
            prevs = wp.previous(step_back)
            if not prevs: break
            wp = prevs[0]; trials += 1
            if (wp.road_id, wp.section_id, wp.lane_id) == target_ids:
                spawn_wp = wp; break

    # —— 3) 生成位姿：车道中心 + 车道朝向；抬高 0.2m 防穿插 —— #
    yaw = float(spawn_wp.transform.rotation.yaw)
    spawn_tf = carla.Transform(
        location=carla.Location(
            x=spawn_wp.transform.location.x,
            y=spawn_wp.transform.location.y,
            z=spawn_wp.transform.location.z + 0.20
        ),
        rotation=carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0)
    )

    # —— 4) 试生成（中心→轻微扰动兜底） —— #
    ego_bp = get_ego_blueprint(world)
    jitters = [(0,0), (0.4,0), (-0.4,0), (0,0.4), (0,-0.4)]
    ego = None
    for dx, dy in jitters:
        rad = math.radians(yaw)
        rx, ry = math.sin(rad), -math.cos(rad)   # right
        fx, fy = math.cos(rad),  math.sin(rad)   # forward
        loc = carla.Location(
            x=spawn_tf.location.x + rx*dx + fx*dy,
            y=spawn_tf.location.y + ry*dx + fy*dy,
            z=spawn_tf.location.z
        )
        try_tf = carla.Transform(loc, spawn_tf.rotation)
        ego = world.try_spawn_actor(ego_bp, try_tf)
        if ego is not None:
            print(
                f"[EGO] 生成 @ lane_center road/section/lane=({spawn_wp.road_id}/{spawn_wp.section_id}/{spawn_wp.lane_id}), "
                f"与锥桶朝向夹角={ang_diff_deg(yaw, yaw_cone):.1f}°，距锥桶上游≈{offset:.2f} m"
            )
            env.set_ego(ego)
            return ego

    raise RuntimeError("未能在目标车道中心附近生成 EGO。")

# ========= 2) 同车道参考线：沿 lane 采样，提供 XY↔(s,ey) =========
class LaneRef:
    """
    沿“同一条驾驶车道”的中心线向前采样，构造参考线：
      - seed_wp：用第一个锥桶所在 lane 的中心作为种子
      - step：采样间距（m）
      - max_len：参考线长度（m）
    提供：
      - xy2se(x,y)   -> (s, ey)
      - se2xy(s,ey) -> (x, y)
      - normal_of_s(s) -> (nx, ny) 左法向
    """
    def __init__(self, amap: carla.Map, seed_wp: carla.Waypoint, step: float = 1.0, max_len: float = 500.0):
        pts = []
        wps = []
        wp = seed_wp
        dist = 0.0
        # 保证始终在同一条 lane（用 road/section/lane_id 守护）
        guard_ids = (wp.road_id, wp.section_id, wp.lane_id)
        pts.append((wp.transform.location.x, wp.transform.location.y))
        wps.append(wp)
        while dist < max_len:
            nxts = wp.next(step)
            if not nxts:
                break
            wp = nxts[0]
            if (wp.road_id, wp.section_id, wp.lane_id) != guard_ids:
                # 碰到路口或切换段就停，避免跨 lane
                break
            pts.append((wp.transform.location.x, wp.transform.location.y))
            wps.append(wp)
            dist += step
        self.P = np.asarray(pts, dtype=float)     # [N,2]
        d = np.linalg.norm(np.diff(self.P, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)])  # [N]
        tang = np.diff(self.P, axis=0)
        tang = np.vstack([tang, tang[-1]])
        self.tang = tang / (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)  # 单位切向
        self.guard_ids = guard_ids
        self.step = step

    def _segment_index_and_t(self, x, y):
        """找最近段并返回（段索引 i，段内 t∈[0,1]，最近点坐标）"""
        P = self.P; T = self.tang
        xy = np.array([x,y], dtype=float)
        v = xy - P[:-1]
        seg = P[1:] - P[:-1]
        seg_len2 = (seg[:,0]**2 + seg[:,1]**2) + 1e-9
        t = np.clip((v[:,0]*seg[:,0] + v[:,1]*seg[:,1]) / seg_len2, 0.0, 1.0)
        proj = P[:-1] + seg * t[:,None]
        dist2 = np.sum((proj - xy[None,:])**2, axis=1)
        i = int(np.argmin(dist2))
        return i, float(t[i]), proj[i]

    def xy2se(self, x: float, y: float):
        i, t, proj = self._segment_index_and_t(x, y)
        s_val = self.s[i] + t * (self.s[i+1] - self.s[i])
        # 左法向：切向旋转 +90°
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        ey = float((x - proj[0]) * nx + (y - proj[1]) * ny)
        return float(s_val), float(ey)

    def se2xy(self, s: float, ey: float):
        """由 s 在折线上找点，再沿左法向偏移 ey"""
        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s)-2))
        ratio = (s - self.s[i]) / max(1e-9, self.s[i+1]-self.s[i])
        base = self.P[i] * (1 - ratio) + self.P[i+1] * ratio
        tx, ty = self.tang[i]
        nx, ny = -ty, tx  # 左法向
        x = base[0] + ey * nx
        y = base[1] + ey * ny
        return float(x), float(y)

    def normal_of_s(self, s: float):
        s = float(np.clip(s, self.s[0], self.s[-1]))
        i = int(np.searchsorted(self.s, s) - 1)
        i = max(0, min(i, len(self.s)-1))
        tx, ty = self.tang[i]
        nx, ny = -ty, tx
        return float(nx), float(ny)


# ========= 3) 规则型 Planner（走廊中线跟随 + 边界防撞） =========
class RuleBasedPlanner:
    """
    - 每隔 dp_interval 帧重建一次走廊
    - 目标横向: ey_ref = (lower+upper)/2 （可加偏置）
    - 边界保护: 接近边界时提高纠偏增益
    - 目标纵向: v_ref 基于走廊宽度/障碍密度做降速
    """
    def __init__(self, dp: DPCorridor, ref: LaneRef,
                 v_ref_base=12.0, dp_interval=8):
        self.dp = dp
        self.ref = ref
        self.v_ref_base = v_ref_base
        self.dp_interval = dp_interval
        self.frame = 0
        self.corridor = None  # type: Corridor|None

    def update_corridor(self, world):
        # 构建 cost-map → DP → corridor
        cost = self.dp.build_cost_map(world, ref_xy2se=self.ref.xy2se, horizon_T=3.0, dt=0.2)
        self.corridor = self.dp.run_dp(cost)

    def _interp_bounds(self, s_now):
        s = self.corridor.s
        lo = np.interp(s_now, s, self.corridor.lower)
        up = np.interp(s_now, s, self.corridor.upper)
        return lo, up

    def compute_control(self, obs: dict, dt: float=0.05) -> tuple[float,float,float,dict]:
        """
        输入: env.get_observation() 输出:
          throttle, steer, brake, debug_info
        规则控制核心：
          - 计算当前 Frenet (s,ey)
          - ey_ref = (lo+up)/2；误差 -> delta
          - 走廊窄/误差大 → 降低 v_ref；纵向 ax 逼近 v_ref
        """
        ego_pose = obs.get("ego_pose", {})
        ego_v    = obs.get("ego_v", {})
        x, y = ego_pose.get("x", 0.0), ego_pose.get("y", 0.0)
        yaw_deg = ego_pose.get("yaw", 0.0)
        speed = float(ego_v.get("speed", 0.0))

        s_now, ey_now = self.ref.xy2se(x, y)

        lo, up = self._interp_bounds(s_now) if self.corridor is not None else (-1.5, 1.5)
        width = max(0.2, up - lo)
        ey_ref = 0.5*(lo + up)  # 走廊中线；可改成偏左/偏右策略
        e_y = ey_ref - ey_now

        # 横向控制：比例 + 边界防撞（靠近边界时增益↑）
        # 基础增益：
        Ky = 0.8
        # 边界权重（越靠边界越大）
        margin = min(abs(ey_now - lo), abs(up - ey_now))
        guard = 1.0 + 0.8*np.exp(-margin/0.3)
        delta_cmd = (Ky*guard) * e_y   # [rad] 近似成前轮转角

        # 纵向参考：走廊越窄，速度越低
        v_ref = max(6.0, min(self.v_ref_base, self.v_ref_base * (width/3.0)))
        # 纵向加速度（简单 P 控制，限制变化率可按需加）
        ax = 0.8*(v_ref - speed)
        ax = np.clip(ax, -3.5, 2.0)

        throttle, brake = ax_to_throttle_brake(ax, speed, a_max=2.0, a_min=-3.5)
        steer = delta_to_steer(delta_cmd, max_steer_rad=0.5)

        dbg = dict(
            mode="DP-corridor + RuleMidline",
            s=s_now, ey=ey_now, lo=lo, up=up, width=width,
            ey_ref=ey_ref, e_y=e_y, guard=guard,
            v=speed, v_ref=v_ref, ax=ax, delta=delta_cmd,
            throttle=throttle, brake=brake, steer=steer,
        )
        return throttle, steer, brake, dbg

    def close(self):
        """清理本环境内登记的所有 actor，并退出同步模式。"""
        try:
            for a in list(self._actors):
                try:
                    a.destroy()
                except Exception:
                    pass
        finally:
            self._actors.clear()
            self.ego = None
            if self._sync_cm is not None:
                # 退出同步
                try:
                    self._sync_cm.__exit__(None, None, None)
                except Exception:
                    pass
                self._sync_cm = None


# ========= 4) 主流程 =======
def main():

    # —— 1) 连接 CARLA & 搭场景 —— #
    env = HighwayEnv(host="127.0.0.1", port=2000, sync=True, fixed_dt=0.05).connect()
    logger = None  # 防止 finally 中 NameError
    try:
        env.setup_scene(
            num_cones=10,
            step_forward=3.0,
            step_right=0.30,
            z_offset=0.0,
            min_gap_from_junction=15.0,
            grid=5.0,
            set_spectator=True,
        )

        # —— 2) 生成 EGO：同车道中心、上游 15–20m —— #
        ego = spawn_ego_upstream_lane_center(env)

        # 相机追尾视角，看清自车
        set_spectator_follow_actor(
            world=env.world,
            actor=ego,
            mode="chase",
            distance=28.0,   # 再往后一些
            height=7.0,
            pitch_deg=-12.0,
            yaw_offset=0.0,
            side_offset=2.0
        )

        # —— 3) 构建同车道参考线（LaneRef） —— #
        first_tf = env.get_first_cone_transform()
        amap = env.world.get_map()
        seed_wp = amap.get_waypoint(first_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ref = LaneRef(amap, seed_wp=seed_wp, step=1.0, max_len=500.0)

        # —— 4) DP 走廊 & 规则 Planner —— #
        dp = DPCorridor(
            s_max=120.0, ds=2.0,
            ey_span=3.0, dey=0.2,
            obs_sigma=0.6, smooth_w=0.05, max_step=2
        )
        planner = RuleBasedPlanner(dp, ref, v_ref_base=12.0, dp_interval=8)

        # —— 5) 诊断日志器（离线 CSV+图） —— #
        logger = TelemetryLogger(out_dir="logs_rule_based")

        dt = 0.05
        frame = 0
        while True:
            # 定期更新 DP 走廊，并在线画出来
            if (planner.corridor is None) or (frame % planner.dp_interval == 0):
                planner.update_corridor(env.world)
                draw_corridor(env.world, ref, planner.corridor)

            # 推进一步仿真 & 计算控制
            obs, _ = env.step()
            throttle, steer, brake, dbg = planner.compute_control(obs, dt=dt)

            # 在线标记自车点（降采样）
            if frame % 2 == 0:
                ego_pose = obs.get("ego_pose", {})
                draw_ego_marker(env.world, ego_pose.get("x", 0.0), ego_pose.get("y", 0.0))

            # 控制信息打印（降采样）
            if frame % 10 == 0:
                print(
                    f"[CTRL] {dbg['mode']} | s={dbg['s']:.1f}, ey={dbg['ey']:.2f}, "
                    f"lo={dbg['lo']:.2f}, up={dbg['up']:.2f}, width={dbg['width']:.2f} "
                    f"| ey_ref={dbg['ey_ref']:.2f}, e_y={dbg['e_y']:.2f}, guard={dbg['guard']:.2f} "
                    f"| v={dbg['v']:.2f} m/s -> v_ref={dbg['v_ref']:.2f} m/s, ax={dbg['ax']:.2f} "
                    f"| delta={dbg['delta']:.3f} rad -> steer={dbg['steer']:.2f} "
                    f"| throttle={dbg['throttle']:.2f}, brake={dbg['brake']:.2f}"
                )
                # 相机跟随也顺便刷新一下
                set_spectator_follow_actor(
                    world=env.world,
                    actor=env.ego,
                    mode="chase",
                    distance=28.0,
                    height=7.0,
                    pitch_deg=-12.0,
                    yaw_offset=0.0,
                    side_offset=2.0
                )

            # 施加控制
            env.apply_control(throttle=throttle, steer=steer, brake=brake)

            # 记录一帧（用于离线图表）
            logger.log(frame, obs, dbg, ref)

            frame += 1

    except KeyboardInterrupt:
        print("\n[Stop] 手动退出。")
    finally:
        if logger is not None:
            logger.save_csv()
            logger.plot()

        planner.close()


if __name__ == "__main__":
    main()
