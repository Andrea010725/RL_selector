from __future__ import annotations
import math
from typing import List, Tuple
import numpy as np
import carla

# ------- 静态障碍物：通过 blueprint type_id 前缀进行白名单筛选 -------
# 你可以按地图资源补充/删减
CONE_PREFIXES = (
    "static.prop.trafficcone",       # 常见命名
    "static.prop.trafficcone01",
    "static.prop.trafficcone02",
    "static.prop.constructioncone",
)
BARRIER_PREFIXES = (
    "static.prop.streetbarrier",
    "static.prop.chainbarrier",
    "static.prop.chainbarrierend",
    "static.prop.barrel",
)
POLE_SIGN_PREFIXES = (
    "static.prop.pole",              # 有些地图会有 pole
    "static.prop.streetsign",
    "static.prop.streetsign01",
    "static.prop.streetsign04",
)

EXCLUDE_ROLE_NAMES = ("hero", "ego")

def _is_static_obstacle_type(tid: str) -> bool:
    tid = tid.lower()
    return (
        any(tid.startswith(p) for p in CONE_PREFIXES) or
        any(tid.startswith(p) for p in BARRIER_PREFIXES) or
        any(tid.startswith(p) for p in POLE_SIGN_PREFIXES)
    )

def _interp(p0, p1, density: float):
    x0, y0 = p0; x1, y1 = p1
    if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
        return
    step = max(0.1, float(density))
    L = float(math.hypot(x1 - x0, y1 - y0))
    if (not np.isfinite(L)) or (L < 1e-8):
        yield (x0, y0); return
    n = max(1, int(L / step))
    for k in range(n + 1):
        t = k / n
        yield (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

def _footprint_from_vertices(vertices_xy: List[Tuple[float,float]], density=0.3):
    densified = []
    if len(vertices_xy) < 4:
        return densified
    for i in range(4):
        a = vertices_xy[i]
        b = vertices_xy[(i + 1) % 4]
        for p in _interp(a, b, density):
            densified.append(p)
    return densified

def _ego_xy_fwd(ego: carla.Actor):
    tf = ego.get_transform()
    ex, ey = tf.location.x, tf.location.y
    yaw = math.radians(tf.rotation.yaw)
    fx, fy = math.cos(yaw), math.sin(yaw)
    return ex, ey, fx, fy

def _ahead_and_near(ex, ey, fx, fy, x, y, r_xy: float, half_fov_cos=math.cos(math.radians(110))):
    dx, dy = x - ex, y - ey
    if dx*dx + dy*dy > r_xy*r_xy:
        return False
    dist = math.hypot(dx, dy) + 1e-9
    cosang = (dx*fx + dy*fy) / dist
    return cosang >= half_fov_cos

def _bb_bottom4_from_actor(a: carla.Actor) -> List[Tuple[float,float]]:
    """优先用 CARLA API 拿 8 顶点→取底面 4 点；失败则用 yaw+extent 回退"""
    try:
        tf = a.get_transform()
        bb = a.bounding_box
    except RuntimeError:
        return []

    # 优先 API
    verts = []
    try:
        verts = bb.get_world_vertices(tf)  # 8 points
    except Exception:
        verts = []
    if verts:
        vv = [(v.x, v.y, v.z) for v in verts if np.isfinite(v.x) and np.isfinite(v.y)]
        if len(vv) >= 4:
            vv.sort(key=lambda p: p[2])
            return [(p[0], p[1]) for p in vv[:4]]

    # 回退：手算四角
    exx, eyy = float(bb.extent.x), float(bb.extent.y)
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    corners_local = [(+exx, +eyy), (+exx, -eyy), (-exx, -eyy), (-exx, +eyy)]
    out = []
    for lx, ly in corners_local:
        wx = tf.location.x + c * lx - s * ly
        wy = tf.location.y + s * lx + c * ly
        if np.isfinite(wx) and np.isfinite(wy):
            out.append((float(wx), float(wy)))
    return out

def collect_obstacles_api(
    world: carla.World,
    ego: carla.Actor,
    ref_xy2se,
    s_center: float,
    s_back: float = 10.0,
    s_fwd: float = 20.0,
    r_xy: float = 35.0,
    horizon_T: float = 2.0,
    dt: float = 0.2,
    static_density: float = 0.3,
) -> List[Tuple[float, float]]:
    """
    仅用 Actors 与其 bounding_box 获取障碍物：
      - 动态：vehicle.*, walker.* （常速外推）
      - 静态：static.prop.* 中的锥桶/路障/路牌杆等（白名单）
    不使用 CityObjectLabel / get_level_bbs；检测阶段不使用 project_to_road。
    """
    pts_se: List[Tuple[float, float]] = []

    # Frenet 窗口
    s_min = float(s_center - s_back)
    s_max = float(s_center + s_fwd)

    # 自车姿态
    ex, ey, fx, fy = _ego_xy_fwd(ego)

    actors = world.get_actors()
    for a in actors:
        # 排除自车 / hero
        if a.id == ego.id:
            continue
        if a.attributes.get("role_name", "") in EXCLUDE_ROLE_NAMES:
            continue

        tid = a.type_id.lower()

        # -------- 动态（车辆/行人） --------
        if tid.startswith("vehicle.") or tid.startswith("walker."):
            try:
                a_tf = a.get_transform()
            except RuntimeError:
                continue
            vel = a.get_velocity()
            spd = float(math.hypot(vel.x, vel.y))

            # 先用包围盒中心做剪枝
            bb_center_world = a_tf.transform(a.bounding_box.location)
            cx, cy = bb_center_world.x, bb_center_world.y
            if not _ahead_and_near(ex, ey, fx, fy, cx, cy, r_xy):
                continue

            base4 = _bb_bottom4_from_actor(a)
            if not base4:
                continue
            fp = _footprint_from_vertices(base4, density=static_density)

            if spd > 0.2:
                steps = max(1, int(horizon_T / max(1e-3, dt)))
                vx, vy = float(vel.x), float(vel.y)
                for (x0, y0) in fp:
                    for k in range(steps + 1):
                        t = k * dt
                        x, y = x0 + vx * t, y0 + vy * t
                        if not _ahead_and_near(ex, ey, fx, fy, x, y, r_xy):
                            continue
                        s, eyy = ref_xy2se(x, y)
                        if np.isfinite(s) and np.isfinite(eyy) and (s_min <= s <= s_max):
                            pts_se.append((float(s), float(eyy)))
            else:
                for (x, y) in fp:
                    if not _ahead_and_near(ex, ey, fx, fy, x, y, r_xy):
                        continue
                    s, eyy = ref_xy2se(x, y)
                    if np.isfinite(s) and np.isfinite(eyy) and (s_min <= s <= s_max):
                        pts_se.append((float(s), float(eyy)))

        # -------- 静态（锥桶/路障/杆类） --------
        elif tid.startswith("static.prop.") and _is_static_obstacle_type(tid):
            base4 = _bb_bottom4_from_actor(a)
            if not base4:
                continue

            # 先用底面中心做剪枝
            cx = sum(p[0] for p in base4) / 4.0
            cy = sum(p[1] for p in base4) / 4.0
            if not _ahead_and_near(ex, ey, fx, fy, cx, cy, r_xy):
                continue

            fp = _footprint_from_vertices(base4, density=static_density)
            for (x, y) in fp:
                s, eyy = ref_xy2se(x, y)
                if np.isfinite(s) and np.isfinite(eyy) and (s_min <= s <= s_max):
                    pts_se.append((float(s), float(eyy)))

        # 其它静态大物体（如建筑、长围栏）一律忽略，避免代价图被淹没

    return pts_se
