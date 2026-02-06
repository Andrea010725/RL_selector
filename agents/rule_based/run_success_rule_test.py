from __future__ import annotations

"""
Rule-Based Agent Success Test:
- 四个场景各跑 30 seeds × 3 trials（与原脚本一致）
- 记录：场景名、episode编号(seed+trial)、是否碰撞
- 若无碰撞，记录 EVA 的 safety / comfort / efficiency（按 episode 均值）

输出：
  /home/ajifang/il_data_collect/il_data/success_rule_report.csv
"""

import sys
import time
import csv
import random
import math
import os
import contextlib
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Any

sys.path.append('/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg')
import carla
sys.path.append("/home/ajifang/RL_selector")

from env.scenarios import (
    ConesScenario,
    JaywalkerScenario,
    TrimmaScenario,
    ConstructionLaneChangeScenario,
)

# 引入你的 rule-based agent
sys.path.append("/home/ajifang/RL_selector/agents/rule_based")
from easy import RuleBasedPlanner

# ================================
# 复制版：carla_utils（不依赖 il_data_collector）
# ================================
@contextlib.contextmanager
def carla_sync_mode(client: carla.Client, world: carla.World, enabled: bool, fixed_dt: float = 0.05):
    if not enabled:
        yield
        return

    original_settings = world.get_settings()
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_dt
        world.apply_settings(settings)

        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(True)
        yield
    finally:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(False)
        world.apply_settings(original_settings)


EGO_BP_CANDIDATES = [
    "vehicle.tesla.model3",
]


def get_ego_blueprint(world: carla.World) -> carla.ActorBlueprint:
    lib = world.get_blueprint_library()
    for name in EGO_BP_CANDIDATES:
        try:
            bp = lib.find(name)
            if bp.has_attribute("color"):
                colors = bp.get_attribute("color").recommended_values
                bp.set_attribute("color", colors[0] if colors else "255,0,0")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
            return bp
        except Exception:
            continue

    for bp in lib.filter("vehicle.*"):
        if bp.has_attribute("number_of_wheels") and bp.get_attribute("number_of_wheels").as_int() == 4:
            if bp.has_attribute("color"):
                bp.set_attribute("color", "255,0,0")
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "hero")
            return bp

    raise RuntimeError("未找到可用车辆蓝图。")


def set_spectator_follow_ego(
    world: carla.World,
    ego: carla.Actor,
    mode: str = "chase",
    distance: float = 22.0,
    height: float = 7.0,
    pitch_deg: float = -12.0,
    side_offset: float = 1.5,
):
    spec = world.get_spectator()
    tf = ego.get_transform()
    yaw = tf.rotation.yaw
    rad = math.radians(yaw)
    fx, fy = math.cos(rad), math.sin(rad)
    rx, ry = math.sin(rad), -math.cos(rad)

    if mode == "top":
        cam_loc = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + height)
        cam_rot = carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0)
    else:
        cam_loc = carla.Location(
            x=tf.location.x - fx * distance + rx * side_offset,
            y=tf.location.y - fy * distance + ry * side_offset,
            z=tf.location.z + height,
        )
        cam_rot = carla.Rotation(pitch=pitch_deg, yaw=yaw, roll=0.0)

    spec.set_transform(carla.Transform(cam_loc, cam_rot))


# ================================
# 复制版：EVA Monitor（不依赖 il_data_collector）
# ================================
try:
    import pygame
except Exception as e:
    raise RuntimeError("需要 pygame 才能显示 EVA HUD，请先安装 pygame") from e


@dataclass
class EVAConfig:
    w_saf: float = 1.0
    w_com: float = 0.2
    w_eff: float = 0.1

    w_col: float = 1.0
    w_dev: float = 0.3
    w_head: float = 0.2
    w_off: float = 0.6
    w_obs: float = 0.6
    w_spd_pen: float = 0.2

    w_steer: float = 0.3
    w_jerk: float = 0.4
    w_brake: float = 0.3

    w_spd: float = 0.6
    w_prog: float = 0.4

    max_dev: float = 2.0
    max_head: float = 0.35
    speed_limit: float = 12.0
    min_speed: float = 0.5
    ttc_crit: float = 3.0
    obs_dist_crit: float = 8.0
    brake_thr: float = 0.5


class EvaMonitor:
    def __init__(self, cfg: EVAConfig = EVAConfig(), width: int = 520, height: int = 320):
        self.cfg = cfg
        self.width = width
        self.height = height
        self.world: Optional[carla.World] = None
        self.ego: Optional[carla.Actor] = None

        self.prev_speed: Optional[float] = None
        self.prev_acc: Optional[float] = None
        self.prev_steer: Optional[float] = None
        self.prev_time: Optional[float] = None
        self.progress_s: float = 0.0

        self.hist_len = 120
        self.hist_total = deque(maxlen=self.hist_len)
        self.hist_speed = deque(maxlen=self.hist_len)
        self.hist_saf = deque(maxlen=self.hist_len)
        self.hist_com = deque(maxlen=self.hist_len)
        self.hist_eff = deque(maxlen=self.hist_len)

        os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "20,20")
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("EVA Monitor")
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_small = pygame.font.SysFont("Arial", 13)
        self.font_micro = pygame.font.SysFont("Arial", 12)

        self.bg = (16, 18, 22)
        self.card = (36, 38, 44)
        self.card_2 = (30, 32, 38)
        self.card_3 = (26, 28, 34)
        self.text = (228, 232, 236)
        self.text_dim = (160, 165, 175)
        self.accent = (255, 156, 66)
        self.accent2 = (96, 192, 255)
        self.good = (112, 212, 140)
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

    def _get_lateral_and_heading_error(self) -> (float, float):
        if self.world is None or self.ego is None:
            return 0.0, 0.0
        amap = self.world.get_map()
        ego_tf = self.ego.get_transform()
        wp = amap.get_waypoint(ego_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            return 0.0, 0.0

        dx = ego_tf.location.x - wp.transform.location.x
        dy = ego_tf.location.y - wp.transform.location.y
        yaw_wp = math.radians(wp.transform.rotation.yaw)
        nx = -math.sin(yaw_wp)
        ny = math.cos(yaw_wp)
        dev = dx * nx + dy * ny

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
                a_vel = a.get_velocity()
                rel_v = ego_speed - math.hypot(a_vel.x, a_vel.y)
                if rel_v > 0.1:
                    ttc = dist / rel_v
                else:
                    ttc = 999.0

        return nearest_dist, ttc

    def _speed_reward(self, v: float) -> float:
        v_des = min(self.cfg.speed_limit, max(self.cfg.min_speed, self.cfg.speed_limit))
        sigma = 2.0
        return math.exp(-((v - v_des) ** 2) / (2 * sigma ** 2))

    def tick(self) -> Dict[str, float]:
        if self.world is None or self.ego is None:
            return {}

        snap = self.world.get_snapshot()
        now = snap.timestamp.elapsed_seconds
        ego_vel = self.ego.get_velocity()
        speed = math.hypot(ego_vel.x, ego_vel.y)

        if self.prev_speed is None or self.prev_time is None:
            acc = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            acc = (speed - self.prev_speed) / dt

        if self.prev_acc is None or self.prev_time is None:
            jerk = 0.0
        else:
            dt = max(1e-3, now - self.prev_time)
            jerk = (acc - self.prev_acc) / dt

        ctrl = self.ego.get_control()
        steer = float(ctrl.steer)

        dev, head_err = self._get_lateral_and_heading_error()
        offroad = 1.0 if self._is_offroad() else 0.0
        obs_dist, ttc = self._nearest_obstacle_info()

        r_col = 0.0
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

        r_steer = -(steer ** 2)
        r_jerk = -min(1.0, abs(jerk) / 5.0)
        r_brake = -max(0.0, (ctrl.brake - self.cfg.brake_thr))

        r_com = (self.cfg.w_steer * r_steer +
                 self.cfg.w_jerk * r_jerk +
                 self.cfg.w_brake * r_brake)

        r_spd = self._speed_reward(speed)
        r_prog = speed / max(1e-3, self.cfg.speed_limit)
        r_eff = self.cfg.w_spd * r_spd + self.cfg.w_prog * r_prog

        r_total = self.cfg.w_saf * r_saf + self.cfg.w_com * r_com + self.cfg.w_eff * r_eff

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

        self.hist_total.append(r_total)
        self.hist_speed.append(speed)
        self.hist_saf.append(r_saf)
        self.hist_com.append(r_com)
        self.hist_eff.append(r_eff)

        return data

    def render(self, data: Optional[Dict[str, float]] = None):
        if data is None:
            data = {}

        self.screen.fill(self.bg)
        panel = pygame.Rect(10, 10, self.width - 20, self.height - 20)
        pygame.draw.rect(self.screen, self.card, panel, border_radius=14)

        title = self.font_title.render("EVA Monitor", True, self.text)
        self.screen.blit(title, (24, 18))
        sub = self.font_small.render("Safety / Comfort / Efficiency Dashboard", True, self.text_dim)
        self.screen.blit(sub, (24, 44))

        left = pygame.Rect(14, 64, 230, 230)
        pygame.draw.rect(self.screen, self.card_2, left, border_radius=12)

        right = pygame.Rect(252, 64, 250, 230)
        pygame.draw.rect(self.screen, self.card_2, right, border_radius=12)

        def draw_bar(label, value, y, color):
            v = max(-1.0, min(1.0, float(value)))
            pygame.draw.rect(self.screen, self.card_3, (28, y + 18, 185, 10), border_radius=6)
            w = int(92 + 92 * v)
            x0 = 28
            pygame.draw.rect(self.screen, color, (x0, y + 18, w, 10), border_radius=6)
            text = self.font_small.render(f"{label}: {value: .3f}", True, self.text)
            self.screen.blit(text, (28, y))
            txt_r = self.font_small.render(f"{value: .3f}", True, self.text_dim)
            self.screen.blit(txt_r, (200, y))

        draw_bar("Total", data.get("r_total", 0.0), 86, self.accent)
        draw_bar("Safety", data.get("r_saf", 0.0), 124, self.good)
        draw_bar("Comfort", data.get("r_com", 0.0), 162, self.accent)
        draw_bar("Efficiency", data.get("r_eff", 0.0), 200, self.accent2)

        def draw_gauge_card(x, y, w, h, title, value, unit, color, vmin, vmax):
            pygame.draw.rect(self.screen, self.card_3, (x, y, w, h), border_radius=10)
            cx, cy = x + 28, y + 26
            radius = 18
            arc_rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
            pygame.draw.arc(self.screen, (70, 74, 82), arc_rect, math.radians(200), math.radians(340), 3)
            v = max(vmin, min(vmax, float(value)))
            ang = (v - vmin) / (vmax - vmin + 1e-6) * 140 + 200
            rad = math.radians(ang)
            px = cx + int((radius - 2) * math.cos(rad))
            py = cy + int((radius - 2) * math.sin(rad))
            pygame.draw.line(self.screen, color, (cx, cy), (px, py), 2)
            pygame.draw.circle(self.screen, color, (cx, cy), 3)
            t = self.font_micro.render(title, True, self.text_dim)
            self.screen.blit(t, (x + 54, y + 6))
            vtxt = self.font.render(f"{value:.2f} {unit}", True, self.text)
            self.screen.blit(vtxt, (x + 54, y + 24))

        draw_gauge_card(20, 238, 108, 40, "Speed", data.get("speed", 0.0), "m/s", self.accent2, 0.0, 20.0)
        draw_gauge_card(132, 238, 108, 40, "Deviation", data.get("dev", 0.0), "m", self.good, 0.0, self.cfg.max_dev)
        draw_gauge_card(20, 282, 108, 40, "Head Error", data.get("head_err", 0.0), "rad", self.warn, 0.0, self.cfg.max_head)
        draw_gauge_card(132, 282, 108, 40, "TTC", data.get("ttc", 0.0), "s", self.accent, 0.0, 10.0)

        def draw_line_chart(rect, series, color, label, y_min=-1.0, y_max=1.0, scale: float = 1.0):
            x, y, w, h = rect
            pygame.draw.rect(self.screen, self.card_3, rect, border_radius=10)
            if len(series) < 2:
                return
            pts = []
            for i, v in enumerate(series):
                vx = x + int(i * (w - 8) / max(1, len(series) - 1)) + 4
                vv = max(y_min, min(y_max, float(v) * scale))
                vy = y + h - int((vv - y_min) / (y_max - y_min + 1e-6) * (h - 8)) - 4
                pts.append((vx, vy))
            pygame.draw.lines(self.screen, color, False, pts, 2)
            if scale != 1.0:
                label = f"{label} x{scale:.0f}"
            txt = self.font_small.render(label, True, self.text)
            self.screen.blit(txt, (x + 8, y + 6))

        draw_line_chart((264, 84, 230, 52), self.hist_total, self.accent, "Total (History)", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 142, 230, 46), self.hist_saf, self.good, "Safety", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 194, 230, 46), self.hist_com, self.warn, "Comfort", -1.0, 1.0, scale=10.0)
        draw_line_chart((264, 246, 230, 46), self.hist_eff, self.accent2, "Efficiency", -1.0, 1.0, scale=10.0)

        pygame.display.flip()

SCENARIOS = ["cones", "jaywalker", "trimma", "construction"]
SEEDS_PER_SCENE = 30
TRIALS_PER_SEED = 3
MAX_STEPS = 600

OUTPUT_CSV = "/home/ajifang/il_data_collect/il_data/success_rule_report.csv"

try:
    import numpy as np
except Exception:
    np = None


def make_scenario(name: str, world: carla.World, amap: carla.Map, tm_port: int):
    class Cfg:
        pass
    cfg = Cfg()
    cfg.tm_port = tm_port
    cfg.enable_traffic_flow = True
    if name == "cones":
        return ConesScenario(world, amap, cfg)
    if name == "jaywalker":
        return JaywalkerScenario(world, amap, cfg)
    if name == "trimma":
        return TrimmaScenario(world, amap, cfg)
    if name == "construction":
        return ConstructionLaneChangeScenario(world, amap, cfg)
    raise ValueError(name)


def _snapshot_actor_ids(world: carla.World) -> set:
    try:
        return set([a.id for a in world.get_actors()])
    except Exception:
        return set()


def _cleanup_new_actors(world: carla.World, before_ids: set):
    try:
        for a in world.get_actors():
            if a.id in before_ids:
                continue
            if (
                a.type_id.startswith("vehicle.")
                or a.type_id.startswith("walker.")
                or a.type_id.startswith("sensor.")
                or a.type_id.startswith("static.prop")
            ):
                try:
                    a.destroy()
                except Exception:
                    pass
    except Exception:
        pass


class CollisionWatcher:
    def __init__(self, world: carla.World, ego: carla.Actor):
        self.world = world
        self.ego = ego
        self.has_collided = False
        self.sensor = None
        self._setup_sensor()

    def _setup_sensor(self):
        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.ego)
        self.sensor.listen(self._on_collision)

    def _on_collision(self, event: carla.CollisionEvent):
        self.has_collided = True

    def destroy(self):
        if self.sensor and self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


def run_one_episode(client: carla.Client, scenario_name: str, tm_port: int) -> Dict:
    world = client.get_world()
    amap = world.get_map()

    # 场景创建
    scenario = make_scenario(scenario_name, world, amap, tm_port)
    if not scenario.setup():
        scenario.cleanup()
        return {"collision": True, "r_saf": None, "r_com": None, "r_eff": None}

    # 生成 ego
    ego_bp = get_ego_blueprint(world)
    spawn_tf = scenario.get_spawn_transform()
    ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        spawn_tf.location.z += 0.5
        ego = world.try_spawn_actor(ego_bp, spawn_tf)
    if ego is None:
        scenario.cleanup()
        return {"collision": True, "r_saf": None, "r_com": None, "r_eff": None}

    # ✅ 预热几帧，避免刚生成交通流就卡在 tick
    try:
        for _ in range(3):
            world.tick()
    except Exception:
        pass

    # spectator
    try:
        set_spectator_follow_ego(world, ego, mode="chase")
    except Exception:
        pass

    # rule-based planner
    planner = RuleBasedPlanner(amap)
    if scenario_name == "cones":
        planner.enable_auto_lane_change = False
    elif scenario_name == "trimma":
        planner.enable_overtake = True
    elif scenario_name == "construction":
        planner.include_all_props_as_cones = True

    # eva
    eva = EvaMonitor()
    eva.attach(world, ego)

    # collision
    collision = CollisionWatcher(world, ego)

    # 统计 EVA
    saf_list: List[float] = []
    com_list: List[float] = []
    eff_list: List[float] = []

    for _ in range(MAX_STEPS):
        world.tick()

        # 场景特殊逻辑
        if isinstance(scenario, JaywalkerScenario):
            scenario.check_and_trigger(ego.get_location())
            scenario.tick_update()

        # 更新规划与控制
        planner.update_corridor(world, ego, s_ahead=35.0, ds=1.0, debug_draw=False)
        throttle, steer, brake, _ = planner.compute_control(ego)
        ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

        data = eva.tick()
        eva.render(data)

        if not collision.has_collided:
            saf_list.append(float(data.get("r_saf", 0.0)))
            com_list.append(float(data.get("r_com", 0.0)))
            eff_list.append(float(data.get("r_eff", 0.0)))

        if collision.has_collided:
            break

    # 清理
    collision.destroy()
    if ego.is_alive:
        ego.destroy()
    scenario.cleanup()

    if collision.has_collided:
        return {"collision": True, "r_saf": None, "r_com": None, "r_eff": None}

    def _avg(arr):
        return sum(arr) / max(1, len(arr))

    return {
        "collision": False,
        "r_saf": _avg(saf_list),
        "r_com": _avg(com_list),
        "r_eff": _avg(eff_list),
    }


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.get_world()

    # ✅ 确保输出目录存在
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario", "seed", "trial", "collision",
                "r_saf", "r_com", "r_eff",
                "seed_bits"
            ],
        )
        writer.writeheader()

        with carla_sync_mode(client, world, enabled=True, fixed_dt=0.05):
            for scene in SCENARIOS:
                for seed in range(SEEDS_PER_SCENE):
                    bits = []
                    for trial in range(1, TRIALS_PER_SEED + 1):
                        print(f"\n=== {scene} | Seed {seed} | Trial {trial}/{TRIALS_PER_SEED} ===")

                        # 固定随机种子
                        random.seed(seed)
                        if np is not None:
                            np.random.seed(seed)

                        before_ids = _snapshot_actor_ids(world)
                        result = run_one_episode(client, scene, tm_port=8000)

                        collided = int(result["collision"])
                        bits.append("1" if collided else "0")

                        writer.writerow({
                            "scenario": scene,
                            "seed": seed,
                            "trial": trial,
                            "collision": collided,
                            "r_saf": "" if result["r_saf"] is None else f"{result['r_saf']:.6f}",
                            "r_com": "" if result["r_com"] is None else f"{result['r_com']:.6f}",
                            "r_eff": "" if result["r_eff"] is None else f"{result['r_eff']:.6f}",
                            "seed_bits": "".join(bits) if trial == TRIALS_PER_SEED else "",
                        })
                        f.flush()

                        _cleanup_new_actors(world, before_ids)
                        time.sleep(0.5)

    print(f"\n✅ 成功写入报告: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
