# agents/rule_based/vis_debug.py
from __future__ import annotations
import os
import csv
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import carla


# 简易取色
C_WHITE   = carla.Color(255,255,255)  # 参考中心/文字
C_BLUE    = carla.Color( 64,180,255)  # 地图墙-左
C_ORANGE  = carla.Color(255,140, 64)  # 地图墙-右
C_CYAN    = carla.Color( 64,255,255)  # 融合后-左
C_YELLOW  = carla.Color(255,235, 64)  # 融合后-右
C_GREEN   = carla.Color(120,255,120)  # 走廊横跨线（当前 s 的左右连接线）
C_RED     = carla.Color(255, 64, 64)  # 动态障碍：当前点/速度箭头
C_PINK    = carla.Color(255,128,200)  # 动态障碍：预测点
C_LIME    = carla.Color(180,255,100)  # 静态障碍：AABB 顶点/盒

# ================== 在线可视化（画在 CARLA 世界） ================== #
def draw_corridor(world: carla.World,
                  ref,             # 需要有 ref.se2xy(s, ey)
                  corridor,        # 需要有 .s, .lower, .upper
                  color_lo=(0,255,0),
                  color_up=(0,255,0),
                  color_mid=(255,255,0),
                  lifetime: float = 0.2):
    """在 CARLA 里把走廊上下边界 & 中线画出来（每次调用生存时长 lifetime 秒）。"""
    if corridor is None:
        return

    s = corridor.s
    lo = corridor.lower
    up = corridor.upper
    mid = 0.5 * (lo + up)

    def _poly(ey_arr, rgb):
        prev = None
        for si, eyi in zip(s, ey_arr):
            x, y = ref.se2xy(float(si), float(eyi))
            # 把 z 放在路面上方一点点
            wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
            z = (wp.transform.location.z if wp else 0.0) + 0.10
            loc = carla.Location(x=x, y=y, z=z)
            if prev is not None:
                world.debug.draw_line(prev, loc, thickness=0.1,
                                      color=carla.Color(*rgb), life_time=lifetime)
            prev = loc

    _poly(lo, color_lo)
    _poly(up, color_up)
    _poly(mid, color_mid)

def draw_ego_marker(world: carla.World,
                    x: float, y: float,
                    text: str = "",
                    color=(0,150,255),
                    lifetime: float = 0.1):
    """画一个自车的位置点与可选文字。"""
    wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
    z = (wp.transform.location.z if wp else 0.0) + 0.30
    loc = carla.Location(x=x, y=y, z=z)
    world.debug.draw_point(loc, size=0.1, color=carla.Color(*color), life_time=lifetime)
    if text:
        world.debug.draw_string(loc, text, draw_shadow=False,
                                color=carla.Color(*color), life_time=lifetime)

def draw_obstacles_samples(world: carla.World, ref, world_points, color=(255, 0, 0), lifetime=0.2):
    """
    world_points: List[(x,y)] 世界坐标的障碍样本点（预测轨迹或静态）
    """
    for (x, y) in world_points:
        wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
        z = (wp.transform.location.z if wp else 0.0) + 0.05
        world.debug.draw_point(carla.Location(x=x, y=y, z=z), size=0.08,
                               color=carla.Color(*color), life_time=lifetime)
        
        
def draw_pts_se(world, ref, pts_se, color=(255,0,0), size=0.08, life=0.6):
    for (s, ey) in pts_se:
        x, y = ref.se2xy(s, ey)
        z = world.get_map().get_waypoint(carla.Location(x=x, y=y), project_to_road=True).transform.location.z + 0.05
        world.debug.draw_point(carla.Location(x=x, y=y, z=z),
                               size=size, color=carla.Color(*color), life_time=life)

def _loc(x, y, z=0.15):
    return carla.Location(x=float(x), y=float(y), z=float(z))

def _se2loc(ref, s, ey, z=0.15):
    x, y = ref.se2xy(float(s), float(ey))
    return _loc(x, y, z)

def draw_corridor_layers(world, ref, s_arr, map_left, map_right, left, right,
                         step_vis=1, z=0.15, life=0.0, persistent=True,
                         draw_span=True):
    """
    分层画走廊：
      - 地图墙（蓝/橙）
      - 融合后走廊（青/黄）
      - 每个 s 的左右连接线（绿）
    """
    dbg = world.debug

    prev_ml = prev_mr = prev_l = prev_r = None
    for k in range(0, len(s_arr), step_vis):
        s  = float(s_arr[k])
        ml = float(map_left[k]);  mr = float(map_right[k])
        l  = float(left[k]);      r  = float(right[k])

        # 地图墙两个点
        ml_loc = _se2loc(ref, s, ml, z)
        mr_loc = _se2loc(ref, s, mr, z)
        # 融合后两个点
        l_loc  = _se2loc(ref, s, l, z)
        r_loc  = _se2loc(ref, s, r, z)

        # 点
        dbg.draw_point(ml_loc, size=0.08, color=C_BLUE,   life_time=life, persistent_lines=persistent)
        dbg.draw_point(mr_loc, size=0.08, color=C_ORANGE, life_time=life, persistent_lines=persistent)
        dbg.draw_point(l_loc,  size=0.08, color=C_CYAN,   life_time=life, persistent_lines=persistent)
        dbg.draw_point(r_loc,  size=0.08, color=C_YELLOW, life_time=life, persistent_lines=persistent)

        # 沿 s 的折线
        if prev_ml is not None:
            dbg.draw_line(prev_ml, ml_loc, thickness=0.03, color=C_BLUE,   life_time=life, persistent_lines=persistent)
            dbg.draw_line(prev_mr, mr_loc, thickness=0.03, color=C_ORANGE, life_time=life, persistent_lines=persistent)
            dbg.draw_line(prev_l,  l_loc,  thickness=0.03, color=C_CYAN,   life_time=life, persistent_lines=persistent)
            dbg.draw_line(prev_r,  r_loc,  thickness=0.03, color=C_YELLOW, life_time=life, persistent_lines=persistent)

        # 每个 s 的横跨线（绿）：用于直观看宽度
        if draw_span:
            dbg.draw_line(l_loc, r_loc, thickness=0.015, color=C_GREEN, life_time=life, persistent_lines=persistent)

        prev_ml, prev_mr, prev_l, prev_r = ml_loc, mr_loc, l_loc, r_loc


def draw_dynamic_actor_debug(world, ref, a, s_min, s_max, in_window_xy, dt=0.2, steps_pred=6, z=0.25):
    """
    画动态障碍：
      - 当前点（红点）
      - 速度箭头（红色箭头）
      - 预测点（粉点）
      - 若投影到走廊窗口内，则在该 s 位置画一条小的“影响 tick 线”
    """
    dbg = world.debug

    tf  = a.get_transform()
    loc = tf.location
    vel = a.get_velocity()

    # 当前点
    dbg.draw_point(_loc(loc.x, loc.y, z), size=0.1, color=C_RED, life_time=0.0, persistent_lines=True)

    # 速度箭头
    v_scale = 0.5  # 视觉缩放
    end = _loc(loc.x + vel.x * v_scale, loc.y + vel.y * v_scale, z)
    dbg.draw_arrow(_loc(loc.x, loc.y, z), end, thickness=0.03, arrow_size=0.1, color=C_RED, life_time=0.0)

    # 当前投影 tick
    ret = in_window_xy(loc.x, loc.y)
    if ret is not None:
        s, ey = ret
        if s_min <= s <= s_max:
            base = _se2loc(ref, s, 0.0, z)
            hit  = _se2loc(ref, s, ey,  z)
            dbg.draw_line(base, hit, thickness=0.02, color=C_RED, life_time=0.0, persistent_lines=True)

    # 预测点
    for j in range(1, steps_pred + 1):
        fx = loc.x + vel.x * (j * dt)
        fy = loc.y + vel.y * (j * dt)
        dbg.draw_point(_loc(fx, fy, z), size=0.06, color=C_PINK, life_time=0.0, persistent_lines=True)
        ret = in_window_xy(fx, fy)
        if ret is not None:
            s, ey = ret
            if s_min <= s <= s_max:
                base = _se2loc(ref, s, 0.0, z)
                hit  = _se2loc(ref, s, ey,  z)
                dbg.draw_line(base, hit, thickness=0.015, color=C_PINK, life_time=0.0, persistent_lines=True)


def draw_static_actor_debug(world, ref, a, in_window_xy, z=0.25):
    """
    画静态障碍：
      - AABB 盒（酸橙绿）
      - 参与投影的底部顶点（酸橙绿点）
      - 若投影到走廊窗口内，画“影响 tick 线”（酸橙绿）
    """
    dbg = world.debug
    tf  = a.get_transform()
    bb  = getattr(a, "bounding_box", None)

    if bb:
        # 盒
        try:
            dbg.draw_box(bb, tf.rotation, thickness=0.03, color=C_LIME, life_time=0.0)
            verts = bb.get_world_vertices(tf)
            # 取底部近似（z 最小的 4 个顶点）
            verts = sorted(verts, key=lambda v: v.z)[:4]
        except Exception:
            verts = [tf.location]

        for v in verts:
            dbg.draw_point(_loc(v.x, v.y, z), size=0.08, color=C_LIME, life_time=0.0, persistent_lines=True)
            ret = in_window_xy(v.x, v.y)
            if ret is None: 
                continue
            s, ey = ret
            base = _se2loc(ref, s, 0.0, z)
            hit  = _se2loc(ref, s, ey,  z)
            dbg.draw_line(base, hit, thickness=0.02, color=C_LIME, life_time=0.0, persistent_lines=True)
    else:
        # 没有 bb 就画位置点
        loc = tf.location
        dbg.draw_point(_loc(loc.x, loc.y, z), size=0.08, color=C_LIME, life_time=0.0, persistent_lines=True)
        ret = in_window_xy(loc.x, loc.y)
        if ret is not None:
            s, ey = ret
            base = _se2loc(ref, s, 0.0, z)
            hit  = _se2loc(ref, s, ey,  z)
            dbg.draw_line(base, hit, thickness=0.02, color=C_LIME, life_time=0.0, persistent_lines=True)

def draw_lane_envelope(world, ref, s_arr, left, right,
                       step_vis=1,                # 每多少个采样点画一个点/线
                       life=0.0,                  # 0 表示只在当前帧；>0 持续秒数
                       persistent=True,           # True 表示持续显示
                       z=0.15):                   # 抬高一点避免与地面重合
    dbg = world.debug

    # 颜色
    C_CENTER = carla.Color(255, 255, 255)  # 白：参考中心
    C_LEFT   = carla.Color( 64, 200, 255)  # 蓝：左边界
    C_RIGHT  = carla.Color(255, 120,  64)  # 橙：右边界
    C_SPAN   = carla.Color(120, 255, 120)  # 绿：跨线(左右连线)

    prev_center = prev_left = prev_right = None

    for k in range(0, len(s_arr), step_vis):
        s = float(s_arr[k])
        eyL = float(left[k])
        eyR = float(right[k])

        # 中心/左右边界在世界系的位置
        cx, cy = ref.se2xy(s, 0.0)
        lx, ly = ref.se2xy(s, eyL)
        rx, ry = ref.se2xy(s, eyR)

        c_loc = _loc(cx, cy, z)
        l_loc = _loc(lx, ly, z)
        r_loc = _loc(rx, ry, z)

        # 画点
        dbg.draw_point(c_loc, size=0.08, color=C_CENTER, life_time=life, persistent_lines=persistent)
        dbg.draw_point(l_loc, size=0.08, color=C_LEFT,   life_time=life, persistent_lines=persistent)
        dbg.draw_point(r_loc, size=0.08, color=C_RIGHT,  life_time=life, persistent_lines=persistent)

        # 画左右边界折线 & 跨线
        if prev_center is not None:
            dbg.draw_line(prev_center, c_loc, thickness=0.03, color=C_CENTER, life_time=life, persistent_lines=persistent)
            dbg.draw_line(prev_left,   l_loc, thickness=0.03, color=C_LEFT,   life_time=life, persistent_lines=persistent)
            dbg.draw_line(prev_right,  r_loc, thickness=0.03, color=C_RIGHT,  life_time=life, persistent_lines=persistent)

        # 在该 s 位置画一条左右连接线（横向宽度直观）
        dbg.draw_line(l_loc, r_loc, thickness=0.015, color=C_SPAN, life_time=life, persistent_lines=persistent)

        prev_center, prev_left, prev_right = c_loc, l_loc, r_loc

def annotate_lane_width(world, ref, s_arr, amap, step_anno=5, z=0.25, life=2.0, persistent=False):
    """每隔 step_anno 个点，标注 s 与 lane_width（用于核对 get_waypoint 的宽度）"""
    dbg = world.debug
    for k in range(0, len(s_arr), step_anno):
        s = float(s_arr[k])
        x, y = ref.se2xy(s, 0.0)
        wp = amap.get_waypoint(carla.Location(x=x, y=y), project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_w = float(wp.lane_width) if wp and getattr(wp, "lane_width", 0.0) else 3.5
        text = f"s={s:.1f}m, w={lane_w:.2f}m"
        dbg.draw_string(carla.Location(x=float(x), y=float(y), z=float(z)),
                        text, draw_shadow=False, color=carla.Color(255,255,0),
                        life_time=life, persistent_lines=persistent)



# ================== 离线记录 & 画图 ================== #
class TelemetryLogger:
    """
    轻量日志器：
      - log(frame, obs, dbg, ref): 记录一帧（用 ref.xy2se 算 s/ey）
      - save_csv(): 保存 CSV
      - plot(): 生成 ey_vs_s / speed / controls 三张图（使用 Agg 后端）
    """
    def __init__(self, out_dir="logs_rule_based"):
        self.data = defaultdict(list)
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def log(self, frame: int, obs: Dict[str, Any], dbg: Dict[str, Any], ref) -> None:
        x = float(obs["ego_pose"]["x"]); y = float(obs["ego_pose"]["y"])
        s_now, ey_now = ref.xy2se(x, y)
        self.data["t"].append(frame)
        self.data["s"].append(s_now)
        self.data["ey"].append(ey_now)
        self.data["lo"].append(float(dbg.get("lo", float("nan"))))
        self.data["up"].append(float(dbg.get("up", float("nan"))))
        self.data["v"].append(float(dbg.get("v", float("nan"))))
        self.data["v_ref"].append(float(dbg.get("v_ref", float("nan"))))
        self.data["ax"].append(float(dbg.get("ax", float("nan"))))
        self.data["delta"].append(float(dbg.get("delta", float("nan"))))
        self.data["throttle"].append(float(dbg.get("throttle", float("nan"))))
        self.data["brake"].append(float(dbg.get("brake", float("nan"))))
        self.data["steer"].append(float(dbg.get("steer", float("nan"))))

    def save_csv(self) -> str:
        path = os.path.join(self.out_dir, "telemetry.csv")
        keys = list(self.data.keys())
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for i in range(len(self.data["t"])):
                w.writerow([self.data[k][i] for k in keys])
        print(f"[LOG] CSV 已保存: {path}")
        return path

    def plot(self) -> None:
        # 避免无显示环境报错，把导入放在函数里
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        # 1) ey vs s with corridor
        s = np.array(self.data["s"]); ey = np.array(self.data["ey"])
        lo = np.array(self.data["lo"]); up = np.array(self.data["up"])
        plt.figure()
        ok = ~np.isnan(lo) & ~np.isnan(up)
        if ok.any():
            plt.plot(s[ok], lo[ok], label="lower", linewidth=1)
            plt.plot(s[ok], up[ok], label="upper", linewidth=1)
            plt.fill_between(s[ok], lo[ok], up[ok], color="#b0e3b0", alpha=0.3, label="corridor")
        plt.plot(s, ey, label="ego ey", linewidth=2)
        plt.xlabel("s (m)"); plt.ylabel("ey (m)"); plt.legend(); plt.grid(True)
        p1 = os.path.join(self.out_dir, "ey_vs_s.png")
        plt.savefig(p1, dpi=150); plt.close(); print(f"[LOG] 图已保存: {p1}")

        # 2) speed
        t = np.array(self.data["t"])
        v = np.array(self.data["v"]); vref = np.array(self.data["v_ref"])
        plt.figure()
        plt.plot(t, v, label="speed (m/s)")
        plt.plot(t, vref, label="v_ref (m/s)")
        plt.xlabel("frame"); plt.ylabel("speed (m/s)"); plt.legend(); plt.grid(True)
        p2 = os.path.join(self.out_dir, "speed.png")
        plt.savefig(p2, dpi=150); plt.close(); print(f"[LOG] 图已保存: {p2}")

        # 3) controls
        ax = np.array(self.data["ax"])
        throttle = np.array(self.data["throttle"])
        brake = np.array(self.data["brake"])
        steer = np.array(self.data["steer"])
        plt.figure()
        plt.plot(t, ax, label="ax (m/s^2)")
        plt.plot(t, throttle, label="throttle")
        plt.plot(t, brake, label="brake")
        plt.plot(t, steer, label="steer")
        plt.xlabel("frame"); plt.legend(); plt.grid(True)
        p3 = os.path.join(self.out_dir, "controls.png")
        plt.savefig(p3, dpi=150); plt.close(); print(f"[LOG] 图已保存: {p3}")


