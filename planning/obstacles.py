# planning/obstacles.py
from __future__ import annotations
from typing import List, Tuple
import math

# 识别静态道具（UE5/0.9.15 常见命名）
STATIC_HINTS = ("static.prop.", "trafficcone", "traffic_cone", "barrier", "warning")

def _alive(a) -> bool:
    try:
        _ = a.id
        return True
    except Exception:
        return False

def _is_dynamic(a) -> bool:
    tid = getattr(a, "type_id", "").lower()
    return tid.startswith("vehicle.") or tid.startswith("walker.")

def _is_static(a) -> bool:
    tid = getattr(a, "type_id", "").lower()
    if tid.startswith("sensor."):  # 过滤传感器
        return False
    return any(k in tid for k in STATIC_HINTS)

def _bb_bottom_vertices_world(actor):
    """返回底面4角世界坐标；API不可用时返回[]。"""
    try:
        bb = actor.bounding_box
        tf = actor.get_transform()
        verts = bb.get_world_vertices(tf)
        if not verts:
            return []
        # 取 z 最小的四个点作为“底面”
        bottom = sorted(verts, key=lambda v: v.z)[:4]
        return [(v.x, v.y) for v in bottom]
    except Exception:
        return []

def collect_obstacles_api(
    world,
    ego,
    ref_xy2se,           # xy -> (s,ey)
    s_center: float,     # 以 Frenet s 为中心（通常是自车 s）
    s_back: float = 0.0, # 自车后向（极简版本默认不看后向）
    s_fwd: float = 20.0, # 自车前向 20 m
    r_xy: float = 60.0,  # 世界半径预筛
    horizon_T: float = 0.0,  # 极简：不预测，填 0 即只取当前
    dt: float = 0.2,
    static_density: float = 0.0,  # 极简：不用插值，留参数不生效
) -> List[Tuple[float, float]]:
    """
    极简障碍采样：在自车前 0–s_fwd m（Frenet 窗口）内，收集：
      - 动态(车辆/行人)：只取中心点（不预测）
      - 静态(道具/锥桶)：取中心点 + 底面四角（若可得）
    返回：[(s,ey), ...]
    """
    pts: List[Tuple[float, float]] = []

    # Frenet s 窗口
    s_min = float(s_center - max(0.0, s_back))
    s_max = float(s_center + max(0.0, s_fwd))

    ego_tf = ego.get_transform()
    ex, ey = ego_tf.location.x, ego_tf.location.y

    actors = world.get_actors()
    for a in actors:
        if not _alive(a) or a.id == ego.id:
            continue

        # 世界坐标欧氏预筛
        tf = a.get_transform()
        ax, ay = tf.location.x, tf.location.y
        dx, dy = ax - ex, ay - ey
        if (dx*dx + dy*dy) > (r_xy * r_xy):
            continue

        # 投 Frenet 看是否落在 s 窗口
        try:
            s0, e0 = ref_xy2se(ax, ay)
        except Exception:
            continue
        if not (s_min <= s0 <= s_max):
            continue

        if _is_dynamic(a):
            # 动态：只用中心点（最稳）
            pts.append((float(s0), float(e0)))
        elif _is_static(a):
            # 静态：中心点 + 底面四角
            pts.append((float(s0), float(e0)))
            for (vx, vy) in _bb_bottom_vertices_world(a):
                try:
                    s1, e1 = ref_xy2se(vx, vy)
                    if s_min <= s1 <= s_max:
                        pts.append((float(s1), float(e1)))
                except Exception:
                    continue
        else:
            # 其他：忽略
            continue

    return pts
