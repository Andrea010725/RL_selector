#!/usr/bin/env python
# 障碍物边界提取和可视化测试脚本
import sys
import math
from typing import List, Tuple
sys.path.append("/home/ajifang/czw/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
import carla
import numpy as np

# 核心的障碍物检测和可视化函数
INCLUDE_PREFIXES = (
    "vehicle.", "walker.", "static.",
    "traffic.", "prop.", "construction.", "barrier."
)
EXCLUDE_ROLE_NAMES = {"hero"}

def _is_candidate(actor: carla.Actor) -> bool:
    """判断是否为障碍物候选"""
    tid = actor.type_id.lower()
    if not any(tid.startswith(p) for p in INCLUDE_PREFIXES):
        # 兜底：多数 actor 都有 bbox；若 bbox 很小或中等，也当作障碍候选
        bb = getattr(actor, "bounding_box", None)
        if not bb:
            return False
        ex, ey = bb.extent.x, bb.extent.y
        return (ex <= 4.0 and ey <= 4.0)
    return True

def _alive(actor: carla.Actor) -> bool:
    """检查actor是否存活"""
    try:
        _ = actor.get_transform()
        return True
    except Exception:
        return False

def _footprint_points(actor: carla.Actor, density: float = 0.5) -> List[Tuple[float, float]]:
    """
    用 actor.bounding_box 生成其在世界坐标系下的 footprint 采样点（间距 density 米）。
    对动态体也可用（当前帧位姿）。
    """
    pts: List[Tuple[float, float]] = []
    bb = getattr(actor, "bounding_box", None)
    tf = actor.get_transform()
    if not bb:
        # 没有 bbox 的（罕见），退化为一个点
        return [(tf.location.x, tf.location.y)]

    # 取包围盒四角（局部坐标）
    ex, ey = bb.extent.x, bb.extent.y
    local_corners = [
        (+ex, +ey, 0.0), (+ex, -ey, 0.0),
        (-ex, -ey, 0.0), (-ex, +ey, 0.0),
    ]
    # 转世界
    yaw = math.radians(tf.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    for (lx, ly, _lz) in local_corners:
        wx = tf.location.x + c*lx - s*ly
        wy = tf.location.y + s*lx + c*ly
        pts.append((wx, wy))
    # 沿四边插点，密度 density
    def _interp(a, b):
        (x1, y1), (x2, y2) = a, b
        L = math.hypot(x2-x1, y2-y1)
        n = max(1, int(L / max(0.1, density)))
        for i in range(n+1):
            t = i / n
            yield (x1 + t*(x2-x1), y1 + t*(y2-y1))
    poly = pts[:]
    densified: List[Tuple[float, float]] = []
    for i in range(4):
        for p in _interp(poly[i], poly[(i+1) % 4]):
            densified.append(p)
    return densified

def collect_obstacles_around_ego(world: carla.World, ego: carla.Actor, r_xy: float = 35.0) -> List[Tuple[float, float]]:
    """
    收集自车周围的所有障碍物点（世界坐标）
    返回：[(x, y), ...]
    """
    ego_tf = ego.get_transform()
    ex, ey = ego_tf.location.x, ego_tf.location.y
    r2 = r_xy * r_xy

    obstacle_points: List[Tuple[float, float]] = []
    actors = world.get_actors()

    print(f"[DEBUG] 检查 {len(actors)} 个actors")
    vehicle_count = 0
    static_count = 0

    for a in actors:
        try:
            if not _alive(a):
                continue
            if a.id == ego.id:
                continue
            role = a.attributes.get("role_name", "")
            if role in EXCLUDE_ROLE_NAMES:
                continue
            if not _is_candidate(a):
                continue

            loc = a.get_transform().location
            dx, dy = loc.x - ex, loc.y - ey
            if dx*dx + dy*dy > r2:
                continue

            tid = a.type_id.lower()
            is_dynamic = tid.startswith("vehicle.") or tid.startswith("walker.")

            print(f"[DEBUG] 发现障碍物: {tid}, 动态: {is_dynamic}, 位置: ({loc.x:.2f}, {loc.y:.2f})")

            if is_dynamic:
                # 动态障碍物：预测轨迹
                vel = a.get_velocity()
                x0, y0 = loc.x, loc.y
                vx, vy = vel.x, vel.y
                # 简化：只取当前位置
                obstacle_points.append((x0, y0))
                vehicle_count += 1
            else:
                # 静态障碍物：用footprint
                footprint = _footprint_points(a, density=0.4)
                obstacle_points.extend(footprint)
                static_count += 1

        except Exception as e:
            print(f"[DEBUG] 处理actor异常: {e}")
            continue

    print(f"[DEBUG] 找到障碍物: {vehicle_count} 动态, {static_count} 静态, 总计 {len(obstacle_points)} 个点")
    return obstacle_points

def draw_obstacle_points(world: carla.World, obstacle_points: List[Tuple[float, float]],
                        color=(255, 0, 0), size=0.1, lifetime=5.0):
    """
    在CARLA中绘制障碍物点
    """
    print(f"[DEBUG] 开始绘制 {len(obstacle_points)} 个障碍物点")
    for i, (x, y) in enumerate(obstacle_points):
        try:
            # 获取路面高度
            wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
            z = (wp.transform.location.z if wp else 0.0) + 0.1  # 抬高0.1米避免Z-fighting

            # 绘制点
            world.debug.draw_point(
                carla.Location(x=x, y=y, z=z),
                size=size,
                color=carla.Color(*color),
                life_time=lifetime
            )

            if i % 10 == 0:  # 每10个点打印一次进度
                print(f"[DEBUG] 已绘制 {i+1}/{len(obstacle_points)} 个点")

        except Exception as e:
            print(f"[DEBUG] 绘制点 ({x:.2f}, {y:.2f}) 失败: {e}")

def test_obstacle_detection():
    """测试障碍物检测和可视化"""
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    try:
        world = client.get_world()
        print("[INFO] 连接到CARLA世界成功")

        # 寻找hero车辆
        ego = None
        for actor in world.get_actors():
            if actor.attributes.get("role_name") == "hero":
                ego = actor
                break

        if ego is None:
            print("[ERROR] 没有找到hero车辆")
            return

        print(f"[INFO] 找到ego车辆: {ego.type_id}")
        ego_loc = ego.get_transform().location
        print(f"[INFO] Ego位置: ({ego_loc.x:.2f}, {ego_loc.y:.2f}, {ego_loc.z:.2f})")

        # 收集障碍物点
        obstacle_points = collect_obstacles_around_ego(world, ego, r_xy=50.0)

        if not obstacle_points:
            print("[WARN] 没有检测到障碍物点")
            return

        # 绘制障碍物点
        draw_obstacle_points(world, obstacle_points, color=(255, 0, 0), size=0.15, lifetime=10.0)
        print(f"[INFO] 已绘制 {len(obstacle_points)} 个障碍物点，持续时间 10 秒")

        # 同时绘制一些更显眼的点用于测试
        ego_x, ego_y = ego_loc.x, ego_loc.y
        test_points = [
            (ego_x + 5, ego_y, "前方5米"),
            (ego_x - 5, ego_y, "后方5米"),
            (ego_x, ego_y + 5, "左侧5米"),
            (ego_x, ego_y - 5, "右侧5米"),
        ]

        for x, y, desc in test_points:
            wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
            z = (wp.transform.location.z if wp else 0.0) + 0.2
            world.debug.draw_point(
                carla.Location(x=x, y=y, z=z),
                size=0.2,
                color=carla.Color(0, 255, 0),  # 绿色测试点
                life_time=15.0
            )
            print(f"[INFO] 绘制测试点: {desc} at ({x:.2f}, {y:.2f}, {z:.2f})")

    except Exception as e:
        print(f"[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_obstacle_detection()