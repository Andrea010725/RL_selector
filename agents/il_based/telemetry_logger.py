# agents/rule_based/telemetry_logger.py
import os
import csv
from typing import Optional, Dict, Any, List

import matplotlib
# 远程/无显示环境自动使用非交互后端
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt


class TelemetryLogger:
    """
    轻量版遥测记录器（扩展版）：
    - 每帧记录：speed, steer, throttle, brake, v_ref（可选）
    - 可选：ey（横向偏移）、s（纵向弧长）、ax（纵向加速度）
    - 导出 CSV + 绘图：
        1) speed/v_ref vs frame           -> speed.png
        2) steer vs frame                 -> steer.png
        3) ey vs frame（若有）            -> ey.png
        4) s  vs frame（若有）            -> s.png
        5) ey vs s（若二者都有）          -> ey_vs_s.png
        6) throttle/brake/steer 合并图    -> controls.png
        7) ax vs frame（若有）            -> ax.png   ← 新增单独输出
    """

    def __init__(self, out_dir: str = "logs_rule_based", csv_name: str = "telemetry.csv"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, csv_name)
        self.rows: List[Dict[str, Any]] = []

    def log(
        self,
        frame: int,
        *,
        speed: float,
        steer: float,
        throttle: float = 0.0,
        brake: float = 0.0,
        v_ref: Optional[float] = None,
        ey: Optional[float] = None,
        s: Optional[float] = None,
        ax: Optional[float] = None,              # 纵向加速度（m/s^2）
        extras: Optional[Dict[str, Any]] = None,
    ):
        row = {
            "frame": int(frame),
            "speed": float(speed),
            "steer": float(steer),
            "throttle": float(throttle),
            "brake": float(brake),
        }
        if v_ref is not None:
            row["v_ref"] = float(v_ref)
        if ey is not None:
            row["ey"] = float(ey)
        if s is not None:
            row["s"] = float(s)
        if ax is not None:
            row["ax"] = float(ax)
        if extras:
            for k, v in extras.items():
                if k not in row:
                    row[k] = v
        self.rows.append(row)

    def save_csv(self):
        if not self.rows:
            print("[TelemetryLogger] 无数据，跳过 CSV 导出。")
            return
        # 汇总所有字段（包含 extras）
        all_keys = []
        for r in self.rows:
            for k in r.keys():
                if k not in all_keys:
                    all_keys.append(k)

        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(self.rows)
        print(f"[TelemetryLogger] CSV 已保存：{self.csv_path}")

    def plot(self):
        if not self.rows:
            print("[TelemetryLogger] 无数据，跳过绘图。")
            return

        # 基础序列
        t = [r["frame"] for r in self.rows]
        v = [r.get("speed", 0.0) for r in self.rows]
        steer = [r.get("steer", 0.0) for r in self.rows]
        v_ref = [r.get("v_ref", None) for r in self.rows]
        has_vref = any(x is not None for x in v_ref)

        # 可选序列
        ey_vals = [r.get("ey", None) for r in self.rows]
        s_vals  = [r.get("s", None)  for r in self.rows]
        ax_vals = [r.get("ax", None) for r in self.rows]
        has_ey = any(x is not None for x in ey_vals)
        has_s  = any(x is not None for x in s_vals)
        has_ax = any(x is not None for x in ax_vals)

        # 速度（含 v_ref）
        plt.figure()
        plt.plot(t, v, label="Speed (m/s)")
        if has_vref:
            v_ref_num = [float("nan") if x is None else x for x in v_ref]
            plt.plot(t, v_ref_num, linestyle="--", label="v_ref (m/s)")
        plt.xlabel("Frame")
        plt.ylabel("Speed (m/s)")
        plt.title("Vehicle Speed")
        plt.legend()
        speed_png = os.path.join(self.out_dir, "speed.png")
        plt.savefig(speed_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[TelemetryLogger] 速度曲线已保存：{speed_png}")

        # 转向
        plt.figure()
        plt.plot(t, steer, label="Steer")
        plt.xlabel("Frame")
        plt.ylabel("Steer")
        plt.title("Steering Command")
        plt.ylim(-1.05, 1.05)  # 若你的 steer 非 -1~1，可调整/删除
        plt.legend()
        steer_png = os.path.join(self.out_dir, "steer.png")
        plt.savefig(steer_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[TelemetryLogger] 转向曲线已保存：{steer_png}")

        # ey vs frame（若有 ey）
        if has_ey:
            plt.figure()
            ey_num = [float("nan") if x is None else x for x in ey_vals]
            plt.plot(t, ey_num, label="ey")
            plt.xlabel("Frame")
            plt.ylabel("ey")
            plt.title("Lateral Offset ey")
            plt.legend()
            ey_png = os.path.join(self.out_dir, "ey.png")
            plt.savefig(ey_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[TelemetryLogger] ey 曲线已保存：{ey_png}")

        # s vs frame（若有 s）
        if has_s:
            plt.figure()
            s_num = [float("nan") if x is None else x for x in s_vals]
            plt.plot(t, s_num, label="s")
            plt.xlabel("Frame")
            plt.ylabel("s")
            plt.title("Longitudinal Position s")
            plt.legend()
            s_png = os.path.join(self.out_dir, "s.png")
            plt.savefig(s_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[TelemetryLogger] s 曲线已保存：{s_png}")

        # ey vs s（若 ey 和 s 都有）
        if has_ey and has_s:
            s_xy, ey_xy = [], []
            for s_v, ey_v in zip(s_vals, ey_vals):
                if (s_v is not None) and (ey_v is not None):
                    s_xy.append(s_v)
                    ey_xy.append(ey_v)
            if s_xy:
                plt.figure()
                plt.plot(s_xy, ey_xy, label="ey(s)")
                plt.xlabel("s")
                plt.ylabel("ey")
                plt.title("ey vs s")
                plt.legend()
                eys_png = os.path.join(self.out_dir, "ey_vs_s.png")
                plt.savefig(eys_png, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"[TelemetryLogger] ey vs s 曲线已保存：{eys_png}")

        # controls：只包含 throttle/brake/steer（去掉 ax）
        plt.figure()
        def _num(arr): return [float("nan") if x is None else x for x in arr]
        thr_num = _num([r.get("throttle", None) for r in self.rows])
        brk_num = _num([r.get("brake", None) for r in self.rows])
        str_num = _num(steer)

        plt.plot(t, thr_num, label="throttle")
        plt.plot(t, brk_num, label="brake")
        plt.plot(t, str_num, label="steer")
        plt.xlabel("Frame")
        plt.title("Controls")
        plt.legend()
        plt.grid(True)
        controls_png = os.path.join(self.out_dir, "controls.png")
        plt.savefig(controls_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[TelemetryLogger] 控制量合并图已保存：{controls_png}")

        # ax 单独绘图（若有）
        if has_ax:
            plt.figure()
            ax_num = [float("nan") if x is None else x for x in ax_vals]
            plt.plot(t, ax_num, label="ax (m/s^2)")
            plt.xlabel("Frame")
            plt.ylabel("ax (m/s^2)")
            plt.title("Longitudinal Acceleration ax")
            plt.legend()
            plt.grid(True)
            ax_png = os.path.join(self.out_dir, "ax.png")
            plt.savefig(ax_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[TelemetryLogger] 纵向加速度图已保存：{ax_png}")
