# agents/rule_based/agent.py
from __future__ import annotations
import math
import random

import sys
sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np
import ipdb


from typing import List, Tuple
sys.path.append("/home/ajifang/czw/RL_selector")
from env.highway_obs import HighwayEnv, get_ego_blueprint
from env.highway_obs import right_unit_vector_from_yaw, forward_unit_vector_from_yaw
from env.highway_obs import shift_location
from planning.dp_corridor import DPCorridor
from planning.obstacles import collect_obstacles_api
from utils.control_mapping import ax_to_throttle_brake, delta_to_steer
from agents.rule_based.vis_debug import draw_corridor, draw_ego_marker, TelemetryLogger, draw_obstacles_samples, draw_pts_se
from env.highway_obs import set_spectator_follow_actor, set_spectator_fixed

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
        self._prev_delta = 0.0
        self._prev_ax = 0.0


    def update_corridor(self, world, ego=None, debug_draw_points: bool = True):
        if ego is None:
            self.corridor = None
            return

        # === 1) 计算本地窗口，并让 DP 的 s_grid 跟随自车滑动 ===
        ego_tf = ego.get_transform()
        s_center, _ = self.ref.xy2se(ego_tf.location.x, ego_tf.location.y)
        s_lo = s_center - 10.0
        s_hi = s_center + 20.0
        # ★ 关键：把 DP 的 s_grid 移到 [s_lo, s_hi]，保证 DP 与采样窗口一致
        self.dp.set_window(s0=s_lo, length_m=(s_hi - s_lo))

        # === 2) 按同一窗口采样通用障碍的 Frenet 点 ===
        pts_se = collect_obstacles_api(
            world=world, ego=ego, ref_xy2se=self.ref.xy2se,
            s_center=s_center, s_back=10.0, s_fwd=20.0,
            r_xy=35.0, horizon_T=2.0, dt=0.2, static_density=0.3
        )

        # （可选）画障碍点：传入 (s,ey)，在函数里 se→xy 再画，避免坐标系混用
        if debug_draw_points and pts_se:
            draw_pts_se(world, self.ref, pts_se, color=(0, 255, 0), size=0.08, life=0.8)

        # === 3) 构造动态障碍 cost_map（软代价，供 DP） ===
        cost = self.dp.build_cost_map_general(
            world=world, ego_actor=ego, ref_xy2se=self.ref.xy2se,
            s_center=s_center, s_back=10.0, s_fwd=20.0,
            sigma_s=2.5, sigma_y=0.6, horizon_T=2.0, dt=0.2
        )
        cost = np.asarray(cost, dtype=float)

        # === 4) 先跑 DP（逐行阈值）得到初始 lo/up ===
        corridor = self.dp.run_dp(cost_map=cost, row_percentile=60.0,
                                  min_width=1.8, safety_margin=0.20)

        # === 5) 用“硬边界墙”钳位 lo/up，让边界贴障碍几何 ===
        if len(pts_se) > 0:
            left_wall, right_wall = self.dp.walls_from_points(pts_se, corridor.s, self.dp.ds, safety=0.25)
            lo = np.maximum(corridor.lower, right_wall)
            up = np.minimum(corridor.upper, left_wall)
            # 最小带宽兜底 + 轻平滑
            bad = (up - lo) < 1.5
            if np.any(bad):
                mid = 0.5 * (up + lo)
                lo[bad] = mid[bad] - 0.75
                up[bad] = mid[bad] + 0.75
            lo = 0.5 * lo + 0.5 * np.r_[lo[:1], lo[:-1]]
            up = 0.5 * up + 0.5 * np.r_[up[:1], up[:-1]]
            corridor.lower, corridor.upper = lo, up

        self.corridor = corridor

    def _interp_bounds(self, s_now):
        s = self.corridor.s
        lo = np.interp(s_now, s, self.corridor.lower)
        up = np.interp(s_now, s, self.corridor.upper)
        return lo, up

    def compute_control(self, obs: dict, dt: float = 0.05) -> tuple[float, float, float, dict]:
        """
        规则控制（修正版）：
          - 坐标对齐：s 夹在 corridor.s 范围内，保证 lo<=up
          - 中线偏置：靠墙时让 ey_ref 向远离墙方向退一点
          - 横向 = e_y + k_heading*e_psi（加航向项），随速衰减；转角限幅+限速
          - 纵向 = 看“宽度”和“最近墙距”同时降速；ax 限幅+限速
        """

        ego_pose = obs.get("ego_pose", {})
        ego_v = obs.get("ego_v", {})
        x, y = float(ego_pose.get("x", 0.0)), float(ego_pose.get("y", 0.0))
        yaw_deg = float(ego_pose.get("yaw", 0.0))
        speed = float(ego_v.get("speed", 0.0))
        yaw_rad = np.deg2rad(yaw_deg)

        # —— 1) 计算 Frenet 位姿 —— #
        s_now, ey_now = self.ref.xy2se(x, y)

        # —— 2) 插值走廊上下界（强制对齐 & 有效性保护） —— #
        if self.corridor is None:
            lo, up = -1.5, 1.5
            s_q = s_now
        else:
            s_min, s_max = float(self.corridor.s[0]), float(self.corridor.s[-1])
            s_q = float(np.clip(s_now, s_min, s_max))  # 夹在范围内
            lo = float(np.interp(s_q, self.corridor.s, self.corridor.lower))
            up = float(np.interp(s_q, self.corridor.s, self.corridor.upper))

        # 保证 lo <= up
        if lo > up:
            lo, up = up, lo

        width = max(1e-3, up - lo)

        # —— 3) 中线 + "靠墙偏置" —— #
        # 距左右墙的有向距离（正值=在边界内，负值=出界）
        margin_r = ey_now - lo        # 距右边界有向距离
        margin_l = up - ey_now        # 距左边界有向距离

        # 靠墙时把目标中线朝"远离墙"偏一点；基于有向距离
        bias_gain = 0.25              # 降低偏置强度（原0.35→0.25）
        bias_zone = 0.40              # 距墙 < 0.4m 进入偏置区
        bias = 0.0

        # 只在车辆在边界内且靠近某一侧时才偏置
        if margin_r < bias_zone and margin_r >= 0 and margin_r <= margin_l:
            bias = +bias_gain * (bias_zone - margin_r)  # 靠近右边界，往左偏
        elif margin_l < bias_zone and margin_l >= 0 and margin_l < margin_r:
            bias = -bias_gain * (bias_zone - margin_l)  # 靠近左边界，往右偏

        ey_ref = 0.5 * (lo + up) + bias
        e_y = ey_ref - ey_now

        # —— 4) 航向误差（基于参考线切线，稳定性改进） —— #
        # 动态调整采样距离，提高数值稳定性
        ds_yaw = max(2.0, speed * 0.3)  # 最小2米，高速时更远
        x_fwd, y_fwd = self.ref.se2xy(s_q + ds_yaw, 0.0)

        # 计算方向向量并检查有效性
        dx, dy = x_fwd - x, y_fwd - y
        if math.hypot(dx, dy) < 0.1:  # 距离太近时使用默认值
            e_psi = 0.0
        else:
            yaw_ref = np.arctan2(dy, dx)  # 参考线方向
            e_psi = np.arctan2(np.sin(yaw_ref - yaw_rad), np.cos(yaw_ref - yaw_rad))  # wrap 到 [-pi,pi]

        # —— 5) 横向控制（e_y + k_psi*e_psi），随速衰减 + guard + 限幅/限速 —— #
        # 更保守的控制增益，避免振荡
        Ky0 = 0.6               # 横向位置增益（原0.9→0.6）
        Kpsi0 = 1.0             # 航向增益（原1.5→1.0）

        # 增强速度衰减，高速时更保守
        Ky = Ky0 / (1.0 + 0.1 * max(0.0, speed))      # 增大衰减因子（原0.05→0.1）
        Kpsi = Kpsi0 / (1.0 + 0.08 * max(0.0, speed)) # 增大衰减因子

        # 边界guard：基于绝对距离，越靠边界增益越大
        min_margin = min(abs(margin_l), abs(margin_r))
        guard = 1.0 + 0.6 * np.exp(-min_margin / 0.25)  # 降低guard强度（原0.8→0.6）

        delta_cmd = guard * (Ky * e_y + Kpsi * e_psi)  # [rad]

        # 更保守的转角限制和变化率限制
        delta_cmd = float(np.clip(delta_cmd, -0.30, 0.30))  # 限制到±17.2°（原±20°）

        if not hasattr(self, "_prev_delta"):
            self._prev_delta = 0.0

        # 降低变化率限制，使转向更平滑
        max_d_delta = np.deg2rad(45.0) * dt  # 变化率上限（原60°→45°/秒）
        d_delta = float(np.clip(delta_cmd - self._prev_delta, -max_d_delta, +max_d_delta))
        delta_cmd = self._prev_delta + d_delta
        self._prev_delta = delta_cmd

        # 改进的转向映射：更保守的最大角度假设
        max_front_wheel_angle = 0.35  # 最大前轮角（原0.5→0.35弧度，约20°）
        steer = float(np.clip(delta_cmd / max_front_wheel_angle, -1.0, 1.0))

        # —— 6) 纵向控制：综合宽度 & 最近墙距降速 —— #
        # 基础目标速度：随“走廊宽度”缩放
        v_base = float(self.v_ref_base)
        v_ref_w = np.clip(v_base * (width / 3.0), 6.0, v_base)
        # 距墙很近再降一档（越近越保守）
        shrink = 1.0
        near = min(margin_l, margin_r)
        if near < 0.6:
            shrink = max(0.4, near / 0.6)  # 0~0.6m → 0.4~1.0
        v_ref = v_ref_w * shrink

        # 加速度（P 控制）+ 限幅 + 变化率限制
        ax_cmd = 0.8 * (v_ref - speed)
        ax_cmd = float(np.clip(ax_cmd, -3.0, 2.0))
        if not hasattr(self, "_prev_ax"):
            self._prev_ax = 0.0
        max_d_ax = 3.0 * dt  # m/s^3，加速度变化率限制
        d_ax = float(np.clip(ax_cmd - self._prev_ax, -max_d_ax, +max_d_ax))
        ax = self._prev_ax + d_ax
        self._prev_ax = ax

        # 踏板映射
        if ax >= 0.0:
            throttle, brake = ax / 2.0, 0.0
        else:
            throttle, brake = 0.0, (-ax) / 3.0
        throttle = float(np.clip(throttle, 0.0, 1.0))
        brake = float(np.clip(brake, 0.0, 1.0))

        dbg = dict(
            mode="DP-corridor + RuleMidline(+bias) + heading [FIXED]",
            s=s_now, s_q=s_q, ey=ey_now, lo=lo, up=up, width=width,
            ey_ref=ey_ref, e_y=e_y, e_psi=e_psi, guard=guard,
            v=speed, v_ref=v_ref, ax=ax, delta=delta_cmd,
            throttle=throttle, brake=brake, steer=steer,
            margin_l=margin_l, margin_r=margin_r, bias=bias
        )
        return throttle, steer, brake, dbg

    def hard_world_cleanup(world: carla.World):
        """
        兜底：扫描并销毁标记为 hero 的自车 & 我们生成的锥桶（不影响其他交通参与者）。
        只有在正常 env.close() 失败或你临时调试时使用。
        """
        try:
            actors = world.get_actors()
            # 销毁 hero（我们设置的 role_name='hero'）
            for v in actors.filter("vehicle.*"):
                try:
                    if v.attributes.get("role_name", "") == "hero":
                        v.destroy()
                except Exception:
                    pass
            # 销毁锥桶
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
        # set_spectator_follow_actor(
        #     world=env.world,
        #     actor=ego,
        #     mode="chase",
        #     distance=28.0,   # 再往后一些
        #     height=7.0,
        #     pitch_deg=-12.0,
        #     yaw_offset=0.0,
        #     side_offset=2.0
        # )
        set_spectator_fixed(world=env.world,
                            ego=ego,
                            back=5.0,       # 你也可用 24~32 之间微调
                            height=7.0,      # 6~9 都可以
                            side_offset=0, # 右偏一点，能看到车身
                            pitch_deg=0.0,
                            look_at_roof=True
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

        # debug - 移除调试断点
        # ipdb.set_trace()
        while True:
            # 定期更新 DP 走廊，并在线画出来
            if (planner.corridor is None) or (frame % planner.dp_interval == 0):
                planner.update_corridor(env.world, ego=env.ego)
                draw_corridor(env.world, ref, planner.corridor)

            # 推进一步仿真 & 计算控制
            obs, _ = env.step()
            throttle, steer, brake, dbg = planner.compute_control(obs, dt=dt)

            # 在线标记自车点（降采样）
            if frame % 2 == 0:
                ego_pose = obs.get("ego_pose", {})
                draw_ego_marker(env.world, ego_pose.get("x", 0.0), ego_pose.get("y", 0.0))

            # 控制信息打印（降采样）+ 转向调试信息
            if frame % 10 == 0:
                print(
                    f"[CTRL] {dbg['mode']} | s={dbg['s']:.1f}, ey={dbg['ey']:.2f}, "
                    f"lo={dbg['lo']:.2f}, up={dbg['up']:.2f}, width={dbg['width']:.2f} "
                    f"| ey_ref={dbg['ey_ref']:.2f}, e_y={dbg['e_y']:.2f}, guard={dbg['guard']:.2f} "
                    f"| v={dbg['v']:.2f} m/s -> v_ref={dbg['v_ref']:.2f} m/s, ax={dbg['ax']:.2f} "
                    f"| delta={dbg['delta']:.3f} rad -> steer={dbg['steer']:.2f} "
                    f"| throttle={dbg['throttle']:.2f}, brake={dbg['brake']:.2f}"
                )
                # 添加转向调试信息
                print(
                    f"[STEER] margin_l={dbg['margin_l']:.3f}, margin_r={dbg['margin_r']:.3f} "
                    f"| e_psi={dbg.get('e_psi', 0.0):.3f}, bias={dbg.get('bias', 0.0):.3f}"
                )
                # # 相机跟随也顺便刷新一下
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
        try:
            if logger is not None:
                logger.save_csv()
                logger.plot()
        except Exception:
            pass
        try:
            env.close()  # ← 只关 env
        except Exception:
            pass
        try:
            planner.hard_world_cleanup(env.world)  # 兜底清理 hero & 锥桶
        except Exception:
            pass


if __name__ == "__main__":
    main()
