# planning/obstacles.py
from __future__ import annotations
from typing import List, Tuple
import math

# 更全的静态关键词（UE5/0.9.15 常见）
STATIC_KEYS = (
    "trafficcone", "traffic_cone",
    "static.prop.trafficcone", "static.prop.trafficcone01", "static.prop.trafficcone02",
    "static.prop.cone",
    "static.prop.trafficwarning", "static.prop.warningconstruction",
    "static.prop.streetbarrier", "static.prop.chainbarrier",
)

EXCLUDE_ROLE_NAMES = ("hero", "ego", "spectator")


def _alive(a) -> bool:
    try:
        _ = a.id
        return True
    except Exception:
        return False


def _is_dynamic(a) -> bool:
    tid = getattr(a, "type_id", "").lower()
    return tid.startswith("vehicle.") or tid.startswith("walker.")


def _is_static_candidate(a) -> bool:
    tid = getattr(a, "type_id", "").lower()
    if tid.startswith("static.prop."):
        return True
    return any(k in tid for k in STATIC_KEYS)


def _footprint_points(actor, density: float = 0.25):
    """
    沿包围盒四边插值采样；小物体同样走边，不再只取中心点。
    density 为“步长（米）”，越小越密。
    """
    import math
    try:
        bb = getattr(actor, "bounding_box", None)
        tf = actor.get_transform()
        if not bb:
            return [(tf.location.x, tf.location.y)]

        ex, ey = float(bb.extent.x), float(bb.extent.y)
        local = [(+ex, +ey, 0.0), (+ex, -ey, 0.0), (-ex, -ey, 0.0), (-ex, +ey, 0.0)]
        yaw = math.radians(tf.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)

        world_corners = []
        for (lx, ly, _lz) in local:
            wx = tf.location.x + c * lx - s * ly
            wy = tf.location.y + s * lx + c * ly
            world_corners.append((wx, wy))

        def _interp(pa, pb, step):
            dx, dy = pb[0] - pa[0], pb[1] - pa[1]
            L = math.hypot(dx, dy)
            n = max(1, int(L / max(1e-3, step)))
            for i in range(n + 1):
                t = i / n
                yield (pa[0] + t * dx, pa[1] + t * dy)

        step = max(0.08, float(density))
        densified = []
        for i in range(4):
            a = world_corners[i]
            b = world_corners[(i + 1) % 4]
            for p in _interp(a, b, step):
                densified.append(p)
        return densified
    except Exception:
        return []


def collect_obstacles_api(
    world,
    ego,
    ref_xy2se,          # xy -> (s, ey)
    s_center: float,
    s_back: float = 10.0,
    s_fwd: float = 20.0,
    r_xy: float = 35.0,
    horizon_T: float = 2.0,
    dt: float = 0.2,
    static_density: float = 0.20,   # 更密一些，边界更连续
) -> List[Tuple[float, float]]:
    """
    输出 Frenet 点 (s, ey)：
      - 动态(vehicle/walker)：常速外推
      - 静态(static.prop.*、trafficcone*、barrier*)：包围盒边缘 footprint 采样
    """
    pts: List[Tuple[float, float]] = []

    s_min = float(s_center - s_back)
    s_max = float(s_center + s_fwd)

    ego_tf = ego.get_transform()
    ex, eyw = ego_tf.location.x, ego_tf.location.y

    # === 动态 ===
    try:
        vehicles = world.get_actors().filter("vehicle.*")
    except Exception:
        vehicles = [a for a in world.get_actors() if _is_dynamic(a)]

    try:
        walkers = world.get_actors().filter("walker.*")
    except Exception:
        walkers = []

    dyns = list(vehicles) + list(walkers)
    steps = max(1, int(horizon_T / max(1e-6, dt)))

    for a in dyns:
        try:
            if not _alive(a):
                continue
            if a.id == ego.id or a.attributes.get("role_name", "") in EXCLUDE_ROLE_NAMES:
                continue

            loc = a.get_transform().location
            dx, dy = loc.x - ex, loc.y - eyw
            if dx * dx + dy * dy > r_xy * r_xy:
                continue

            vel = a.get_velocity()
            x0, y0, vx, vy = loc.x, loc.y, vel.x, vel.y
            for k in range(steps + 1):
                t = k * dt
                x, y = x0 + vx * t, y0 + vy * t
                s, eyf = ref_xy2se(x, y)
                if s_min <= s <= s_max:
                    pts.append((float(s), float(eyf)))
        except Exception:
            continue

    # === 静态（含锥桶/路障） ===
    try:
        statics = world.get_actors().filter("static.prop.*")
    except Exception:
        statics = [a for a in world.get_actors() if _is_static_candidate(a)]

    for a in statics:
        try:
            if not _alive(a):
                continue
            if a.id == ego.id or a.attributes.get("role_name", "") in EXCLUDE_ROLE_NAMES:
                continue

            tid = getattr(a, "type_id", "").lower()
            if (not tid.startswith("static.prop.")) and (not any(k in tid for k in STATIC_KEYS)):
                continue

            loc = a.get_transform().location
            dx, dy = loc.x - ex, loc.y - eyw
            if dx * dx + dy * dy > r_xy * r_xy:
                continue

            # 一律走 footprint（小物体也走边，比中心点稳得多）
            for (x, y) in _footprint_points(a, density=static_density):
                s, eyf = ref_xy2se(x, y)
                if s_min <= s <= s_max:
                    pts.append((float(s), float(eyf)))
        except Exception:
            continue

    return pts

