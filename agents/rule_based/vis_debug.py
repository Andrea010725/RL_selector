# agents/rule_based/vis_debug.py
from __future__ import annotations
import os
import csv
from collections import defaultdict
from typing import Dict, Any

import numpy as np
import carla

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
