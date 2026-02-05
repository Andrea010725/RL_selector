from __future__ import annotations

"""
EVA 评价可视化组件（独立文件）
用途：
1) 实时计算 EVA reward 组成项
2) 以高级 HUD 样式可视化（pygame）

使用方式：
from eva_monitor import EvaMonitor
eva = EvaMonitor()
eva.attach(world, ego)
每帧调用：
eva.tick()
eva.render()

注意：
- 需要 pygame（Carla 常用依赖）
- 需要 CARLA 服务已启动
"""

import math
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from collections import deque

import carla

try:
    import pygame
except Exception as e:
    raise RuntimeError("需要 pygame 才能显示 EVA HUD，请先安装 pygame") from e


# ================================
# 参数配置
# ================================
@dataclass
class EVAConfig:
    # EVA 顶层权重
    w_saf: float = 1.0
    w_com: float = 0.2
    w_eff: float = 0.1

    # Safety 子项权重
    w_col: float = 1.0
    w_dev: float = 0.3
    w_head: float = 0.2
    w_off: float = 0.6
    w_obs: float = 0.6
    w_spd_pen: float = 0.2

    # Comfort 子项权重
    w_steer: float = 0.3
    w_jerk: float = 0.4
    w_brake: float = 0.3

    # Efficiency 子项权重
    w_spd: float = 0.6
    w_prog: float = 0.4

    # 约束阈值
    max_dev: float = 2.0          # 最大允许横向偏差（m）
    max_head: float = 0.35        # 最大允许航向误差（rad）
    speed_limit: float = 12.0     # 期望速度上限（m/s）
    min_speed: float = 0.5        # 最小速度
    ttc_crit: float = 3.0         # TTC 临界值（s）
    obs_dist_crit: float = 8.0    # 近障距离（m）
    brake_thr: float = 0.5        # 刹车舒适阈值


# ================================
# EVA Monitor 主类
# ================================
class EvaMonitor:
    def __init__(self, cfg: EVAConfig = EVAConfig(), width: int = 520, height: int = 320):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.world: Optional[carla.World] = None
        self.ego: Optional[carla.Actor] = None

        # 状态缓存
        self.prev_speed: Optional[float] = None
        self.prev_acc: Optional[float] = None
        self.prev_steer: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.progress_s: float = 0.0

        # 历史序列（用于曲线图）
        self.hist_len = 120
        self.hist_total = deque(maxlen=self.hist_len)
        self.hist_speed = deque(maxlen=self.hist_len)
        self.hist_saf = deque(maxlen=self.hist_len)
        self.hist_com = deque(maxlen=self.hist_len)
        self.hist_eff = deque(maxlen=self.hist_len)

        # pygame 初始化（窗口放左上角）
        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "20,20")
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("EVA Monitor")
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 13)
        self.font_micro = pygame.font.SysFont("Arial", 12)

        # 主题颜色
        self.bg = (16, 18, 22)
        self.card = (36, 38, 44)
        self.card_2 = (30, 32, 38)
        self.card_3 = (26, 28, 34)
        self.text = (228, 232, 236)
        self.text_dim = (160, 165, 175)
        self.accent = (255, 156, 66)   # Total / Comfort
        self.accent2 = (96, 192, 255)  # Efficiency
        self.good = (112, 212, 140)    # Safety
        self.warn = (255, 178, 77)
        self.bad = (219, 88, 96)

    def attach(self, world: carla.World, ego: carla.Actor):
        self.world = world
        self.ego = ego
        self.prev_speed = None
        self.prev_acc = None
        self.prev_steer = None
        self.prev_time = None
        self.progress_s = 0.0
        self.hist_total.clear()
        self.hist_speed.clear()
        self.hist_saf.clear()
        self.hist_com.clear()
        self.hist_eff.clear()

    # ----------------------------
    # 核心指标计算
    # ----------------------------
    def _get_lateral_and_heading_error(self) -> (float, float):
        """
        计算横向偏差与航向误差
        - 横向偏差：ego 到最近车道中心线的偏移
        - 航向误差：ego yaw 与车道切向的角度差
        """
        if self.world is None or self.ego is None:
            return 0.0, 0.0
        amap = self.world.get_map()
        ego_tf = self.ego.get_transform()
        wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0, 0.0

        # 横向偏差
        dx = ego_tf.location.x - wp.transform.location.x
        dy = ego_tf.location.y - wp.transform.location.y
        # 车道切向
        yaw_wp = math.radians(wp.transform.rotation.yaw)
        nx = -math.sin(yaw_wp)
        ny = math.cos(yaw_wp)
        dev = dx * nx + dy * ny

        # 航向误差
        yaw_ego = math.radians(ego_tf.rotation.yaw)
        head_err = (yaw_ego - yaw_wp + math.pi) % (2 * math.pi) - math.pi

        return float(dev), float(head_err)

    def _is_offroad(self) -> bool:
        if self.world is None or self.ego is None:
            return False
        amap = self.world.get_map()
        wp = amap.get_waypoint(self.ego.get_location(), project_to_road=False, lane_type=carla.LaneType.Driving)
        return wp is None

    def _nearest_obstacle_info(self) -> (float, float):
        """
        返回最近动态障碍物的距离和 TTC
        TTC = dist / relative_speed (仅前方)
        """
        if self.world is None or self.ego is None:
            return 999.0, 999.0

        ego_tf = self.ego.get_transform()
        ego_vel = self.ego.get_velocity()
        ego_speed = math.hypot(ego_vel.x, ego_vel.y)
        fwd = ego_tf.get_forward_vector()

        nearest_dist = 999.0
        ttc = 999.0

        actors = self.world.get_actors().filter("vehicle.*|walker.pedestrian.*")
        for a in actors:
            if a.id == self.ego.id:
                continue
            loc = a.get_location()
            dx = loc.x - ego_tf.location.x
            dy = loc.y - ego_tf.location.y
            ahead = dx * fwd.x + dy * fwd.y
            if ahead <= 0:
                continue
            dist = math.hypot(dx, dy)
            if dist < nearest_dist:
                nearest_dist = dist
                # 相对速度估计
                a_vel = a.get_velocity()
                rel_v = ego_speed - math.hypot(a_vel.x, a_vel.y)
                if rel_v > 0.1:
                    ttc = dist / rel_v
                else:
                    ttc = 999.0

        return nearest_dist, ttc

    def _speed_reward(self, v: float) -> float:
        # 速度控制：高斯型（越接近 v_des 越好）
        v_des = min(self.cfg.speed_limit, max(self.cfg.min_speed, self.cfg.speed_limit))
        sigma = 2.0
        return math.exp(-((v - v_des) ** 2) / (2 * sigma ** 2))

    def tick(self) -> Dict[str, float]:
        """
        计算 EVA 组成项并返回 dict
        """
        if self.world is None or self.ego is None:
            return {}

        # 时间与速度
        snap = self.world.get_snapshot()
        now = snap.timestamp.elapsed_seconds
        ego_vel = self.ego.get_velocity()
        speed = math.hypot(ego_vel.x, ego_vel.y)

        # 加速度
        if self.prev_speed is None or self.prev_time is None:
            acc = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            acc = (speed - self.prev_speed) / dt

        # jerk
        if self.prev_acc is None or self.prev_time is None:
            jerk = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            jerk = (acc - self.prev_acc) / dt

        # steering
        ctrl = self.ego.get_control()
        steer = float(ctrl.steer)

        # 横向与航向
        dev, head_err = self._get_lateral_and_heading_error()
        offroad = 1.0 if self._is_offroad() else 0.0
        obs_dist, ttc = self._nearest_obstacle_info()

        # Safety 子项
        r_col = 0.0  # 碰撞由外部检测，这里先置 0
        r_dev = -min(abs(dev) / self.cfg.max_dev, 1.0)
        r_head = -min(abs(head_err) / self.cfg.max_head, 1.0)
        r_off = -offroad
        r_obs = -1.0 if (obs_dist < self.cfg.obs_dist_crit or ttc < self.cfg.ttc_crit) else 0.0
        r_spd_pen = -max(0.0, (speed - self.cfg.speed_limit) / max(1e-3, self.cfg.speed_limit))

        r_saf = (self.cfg.w_col * r_col +
                 self.cfg.w_dev * r_dev +
                 self.cfg.w_head * r_head +
                 self.cfg.w_off * r_off +
                 self.cfg.w_obs * r_obs +
                 self.cfg.w_spd_pen * r_spd_pen)

        # Comfort 子项
        r_steer = -(steer ** 2)
        r_jerk = -min(1.0, abs(jerk) / 5.0)
        r_brake = -max(0.0, (ctrl.brake - self.cfg.brake_thr))

        r_com = (self.cfg.w_steer * r_steer +
                 self.cfg.w_jerk * r_jerk +
                 self.cfg.w_brake * r_brake)

        # Efficiency 子项
        r_spd = self._speed_reward(speed)
        # progress 简化为速度正相关
        r_prog = speed / max(1e-3, self.cfg.speed_limit)
        r_eff = self.cfg.w_spd * r_spd + self.cfg.w_prog * r_prog

        # 总 EVA
        r_total = self.cfg.w_saf * r_saf + self.cfg.w_com * r_com + self.cfg.w_eff * r_eff

        # 更新历史
        self.prev_speed = speed
        self.prev_acc = acc
        self.prev_steer = steer
        self.prev_time = now

        data = {
            "r_total": r_total,
            "r_saf": r_saf,
            "r_com": r_com,
            "r_eff": r_eff,
            "r_dev": r_dev,
            "r_head": r_head,
            "r_off": r_off,
            "r_obs": r_obs,
            "r_spd_pen": r_spd_pen,
            "r_steer": r_steer,
            "r_jerk": r_jerk,
            "r_brake": r_brake,
            "r_spd": r_spd,
            "r_prog": r_prog,
            "speed": speed,
            "dev": dev,
            "head_err": head_err,
            "ttc": ttc,
            "obs_dist": obs_dist,
        }

        # 更新历史序列
        self.hist_total.append(r_total)
        self.hist_speed.append(speed)
        self.hist_saf.append(r_saf)
        self.hist_com.append(r_com)
        self.hist_eff.append(r_eff)

        return data

    # ----------------------------
    # HUD 可视化
    # ----------------------------
    def render(self, data: Optional[Dict[str, float]] = None):
        if data is None:
            data = {}

        # 背景与主卡片（左上角小面板）
        self.screen.fill(self.bg)
        panel = pygame.Rect(10, 10, self.width - 20, self.height - 20)
        pygame.draw.rect(self.screen, self.card, panel, border_radius=14)

        # 标题区
        title = self.font_title.render("EVA Monitor", True, self.text)
        self.screen.blit(title, (24, 18))
        sub = self.font_small.render("Safety / Comfort / Efficiency Dashboard", True, self.text_dim)
        self.screen.blit(sub, (24, 44))

        # 左侧评分卡片
        left = pygame.Rect(14, 64, 230, 230)
        pygame.draw.rect(self.screen, self.card_2, left, border_radius=12)

        # 右侧曲线卡片
        right = pygame.Rect(252, 64, 250, 230)
        pygame.draw.rect(self.screen, self.card_2, right, border_radius=12)

        # 分数条
        def draw_bar(label, value, y, color):
            # value 归一化显示
            v = max(-1.0, min(1.0, float(value)))
            # 进度条背景
            pygame.draw.rect(self.screen, self.card_3, (28, y + 18, 185, 10), border_radius=6)
            # 进度条前景
            w = int(92 + 92 * v)
            x0 = 28
            pygame.draw.rect(self.screen, color, (x0, y + 18, w, 10), border_radius=6)
            text = self.font_small.render(f"{label}: {value: .3f}", True, self.text)
            self.screen.blit(text, (28, y))
            # 右侧数值
            txt_r = self.font_small.render(f"{value: .3f}", True, self.text_dim)
            self.screen.blit(txt_r, (200, y))

        draw_bar("Total", data.get("r_total", 0.0), 86, self.accent)
        draw_bar("Safety", data.get("r_saf", 0.0), 124, self.good)
        draw_bar("Comfort", data.get("r_com", 0.0), 162, self.accent)
        draw_bar("Efficiency", data.get("r_eff", 0.0), 200, self.accent2)

        # 左侧指标卡片（仪表盘感）
        def draw_gauge_card(x, y, w, h, title, value, unit, color, vmin, vmax):
            pygame.draw.rect(self.screen, self.card_3, (x, y, w, h), border_radius=10)
            # 仪表盘圆弧
            cx, cy = x + 28, y + 26
            radius = 18
            arc_rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
            pygame.draw.arc(self.screen, (70, 74, 82), arc_rect, math.radians(200), math.radians(340), 3)
            # 指针
            v = max(vmin, min(vmax, float(value)))
            ang = (v - vmin) / (vmax - vmin + 1e-6) * 140 + 200  # 200~340 度
            rad = math.radians(ang)
            px = cx + int((radius - 2) * math.cos(rad))
            py = cy + int((radius - 2) * math.sin(rad))
            pygame.draw.line(self.screen, color, (cx, cy), (px, py), 2)
            pygame.draw.circle(self.screen, color, (cx, cy), 3)
            # 文本
            t = self.font_micro.render(title, True, self.text_dim)
            self.screen.blit(t, (x + 54, y + 6))
            vtxt = self.font.render(f"{value:.2f} {unit}", True, self.text)
            self.screen.blit(vtxt, (x + 54, y + 24))

        draw_gauge_card(20, 238, 108, 40, "Speed", data.get("speed", 0.0), "m/s", self.accent2, 0.0, 20.0)
        draw_gauge_card(132, 238, 108, 40, "Deviation", data.get("dev", 0.0), "m", self.good, 0.0, self.cfg.max_dev)
        draw_gauge_card(20, 282, 108, 40, "Head Error", data.get("head_err", 0.0), "rad", self.warn, 0.0, self.cfg.max_head)
        draw_gauge_card(132, 282, 108, 40, "TTC", data.get("ttc", 0.0), "s", self.accent, 0.0, 10.0)

        # 右侧曲线图
        def draw_line_chart(rect, series, color, label, y_min=-1.0, y_max=1.0, scale: float = 1.0):
            x, y, w, h = rect
            # 背景
            pygame.draw.rect(self.screen, self.card_3, rect, border_radius=10)
            if len(series) < 2:
                return
            # 归一化
            pts = []
            for i, v in enumerate(series):
                vx = x + int(i * (w - 8) / max(1, len(series) - 1)) + 4
                vv = max(y_min, min(y_max, float(v) * scale))
                vy = y + h - int((vv - y_min) / (y_max - y_min + 1e-6) * (h - 8)) - 4
                pts.append((vx, vy))
            pygame.draw.lines(self.screen, color, False, pts, 2)
            # 标题
            if scale != 1.0:
                label = f"{label} x{scale:.0f}"
            txt = self.font_small.render(label, True, self.text)
            self.screen.blit(txt, (x + 8, y + 6))

        # 右侧曲线（四条卡片）
        draw_line_chart((264, 84, 230, 52), self.hist_total, self.accent, "Total (History)", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 142, 230, 46), self.hist_saf, self.good, "Safety", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 194, 230, 46), self.hist_com, self.warn, "Comfort", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 246, 230, 46), self.hist_eff, self.accent2, "Efficiency", -1.0, 1.0, scale=10.0)

        pygame.display.flip()


# ================================
# 独立运行入口（调试）
# ================================
def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()

    # 自动寻找 ego（role_name=hero）
    ego = None
    for actor in world.get_actors():
        if actor.attributes.get("role_name") in ["hero", "ego"]:
            ego = actor
            break
    if ego is None:
        raise RuntimeError("未找到 role_name=hero 的 Ego")

    eva = EvaMonitor()
    eva.attach(world, ego)

    while True:
        world.tick()
        data = eva.tick()
        eva.render(data)


if __name__ == "__main__":
    main()
