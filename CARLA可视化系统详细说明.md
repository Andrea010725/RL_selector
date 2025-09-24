# CARLA 可视化系统详细说明文档

## 概述

本文档详细介绍repo中所有涉及在CARLA仿真器中绘制图形的代码，包括点、线、文字等可视化元素，以及它们的具体用途和参数设置。

## 可视化系统架构

```
┌─────────────────────────────┐
│        CARLA World          │
│     (carla.World.debug)     │
└─────────────────────────────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
┌───────┐  ┌───────┐  ┌───────┐
│ 画点  │  │ 画线  │  │ 画字  │
│ Point │  │ Line  │  │ Text  │
└───────┘  └───────┘  └───────┘
```

---

## 1. 核心可视化文件：`agents/rule_based/vis_debug.py`

### 1.1 走廊边界绘制：`draw_corridor`

**功能**：绘制路径规划的安全走廊边界线

```python
def draw_corridor(world: carla.World,
                  ref,             # 参考线对象，提供se2xy坐标转换
                  corridor,        # 走廊对象，包含s, lower, upper数据
                  color_lo=(0,255,0),      # 下边界颜色：绿色
                  color_up=(0,255,0),      # 上边界颜色：绿色
                  color_mid=(255,255,0),   # 中线颜色：黄色
                  lifetime: float = 0.2):  # 显示时长(秒)
```

**绘制内容**：
- 🟢 **下边界线**：绿色线条，表示安全走廊的右侧边界
- 🟢 **上边界线**：绿色线条，表示安全走廊的左侧边界
- 🟡 **中线**：黄色线条，表示推荐的行驶路径

**技术细节**：
```python
def _poly(ey_arr, rgb):
    prev = None
    for si, eyi in zip(s, ey_arr):
        # 将Frenet坐标(s,ey)转换为世界坐标(x,y)
        x, y = ref.se2xy(float(si), float(eyi))

        # 获取路面高度，抬高0.1米避免与路面重叠
        wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
        z = (wp.transform.location.z if wp else 0.0) + 0.10
        loc = carla.Location(x=x, y=y, z=z)

        if prev is not None:
            # 使用CARLA的debug.draw_line绘制线段
            world.debug.draw_line(prev, loc, thickness=0.1,
                                  color=carla.Color(*rgb), life_time=lifetime)
        prev = loc
```

**调用时机**：每当走廊更新时调用，帮助可视化车辆的可行区域

---

### 1.2 自车位置标记：`draw_ego_marker`

**功能**：绘制自车当前位置的标记点和可选文字

```python
def draw_ego_marker(world: carla.World,
                    x: float, y: float,        # 自车世界坐标
                    text: str = "",            # 可选显示文字
                    color=(0,150,255),         # 标记颜色：蓝色
                    lifetime: float = 0.1):    # 显示时长(秒)
```

**绘制内容**：
- 🔵 **自车位置点**：蓝色圆点，标记自车当前位置
- 📝 **可选文字**：在点旁边显示调试信息

**技术细节**：
```python
# 获取路面高度，抬高0.3米使其更显眼
wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
z = (wp.transform.location.z if wp else 0.0) + 0.30
loc = carla.Location(x=x, y=y, z=z)

# 绘制点
world.debug.draw_point(loc, size=0.1, color=carla.Color(*color), life_time=lifetime)

# 绘制文字(如果提供)
if text:
    world.debug.draw_string(loc, text, draw_shadow=False,
                            color=carla.Color(*color), life_time=lifetime)
```

**调用时机**：在主循环中定期调用，实时显示自车位置轨迹

---

### 1.3 障碍物采样点绘制：`draw_obstacles_samples`

**功能**：绘制检测到的障碍物采样点（世界坐标）

```python
def draw_obstacles_samples(world: carla.World,
                          ref,                           # 参考线对象（未使用）
                          world_points,                  # List[(x,y)] 世界坐标点
                          color=(255, 0, 0),            # 颜色：红色
                          lifetime=0.2):                # 显示时长(秒)
```

**绘制内容**：
- 🔴 **障碍物点**：红色小点，表示检测到的障碍物位置

**技术细节**：
```python
for (x, y) in world_points:
    # 获取路面高度，轻微抬高避免Z-fighting
    wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
    z = (wp.transform.location.z if wp else 0.0) + 0.05

    # 绘制小红点
    world.debug.draw_point(carla.Location(x=x, y=y, z=z),
                          size=0.08,                      # 较小的点
                          color=carla.Color(*color),
                          life_time=lifetime)
```

**使用场景**：用于调试障碍物检测算法，可视化检测结果

---

### 1.4 Frenet坐标障碍点绘制：`draw_pts_se`

**功能**：绘制Frenet坐标系下的障碍物点

```python
def draw_pts_se(world,
               ref,                    # 参考线对象，提供se2xy转换
               pts_se,                 # List[(s,ey)] Frenet坐标点
               color=(255,0,0),        # 颜色：红色
               size=0.08,              # 点大小
               life=0.6):              # 显示时长(秒)
```

**绘制内容**：
- 🔴 **Frenet障碍点**：红色点，显示Frenet坐标系下的障碍物

**技术细节**：
```python
for (s, ey) in pts_se:
    # 将Frenet坐标转换为世界坐标
    x, y = ref.se2xy(s, ey)

    # 获取路面高度
    wp = world.get_map().get_waypoint(carla.Location(x=x, y=y), project_to_road=True)
    z = wp.transform.location.z + 0.05

    # 绘制点
    world.debug.draw_point(carla.Location(x=x, y=y, z=z),
                          size=size,
                          color=carla.Color(*color),
                          life_time=life)
```

**调用时机**：在`update_corridor`函数中被调用，用于调试Frenet坐标系下的障碍物检测

---

## 2. 测试文件：`test_obstacle_visualization.py`

### 2.1 障碍物点绘制：`draw_obstacle_points`

**功能**：独立的障碍物可视化测试函数

```python
def draw_obstacle_points(world: carla.World,
                        obstacle_points: List[Tuple[float, float]],
                        color=(255, 0, 0),     # 颜色：红色
                        size=0.1,              # 点大小：较大
                        lifetime=5.0):         # 显示时长：5秒
```

**绘制内容**：
- 🔴 **测试障碍点**：用于测试和调试的障碍物可视化

**技术特点**：
- 包含详细的调试输出
- 错误处理机制
- 进度显示（每10个点打印一次）

**技术细节**：
```python
print(f"[DEBUG] 开始绘制 {len(obstacle_points)} 个障碍物点")
for i, (x, y) in enumerate(obstacle_points):
    try:
        # 获取路面高度，抬高0.1米
        wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
        z = (wp.transform.location.z if wp else 0.0) + 0.1

        # 绘制点
        world.debug.draw_point(
            carla.Location(x=x, y=y, z=z),
            size=size,
            color=carla.Color(*color),
            life_time=lifetime
        )

        # 进度显示
        if i % 10 == 0:
            print(f"[DEBUG] 已绘制 {i+1}/{len(obstacle_points)} 个点")

    except Exception as e:
        print(f"[DEBUG] 绘制点 ({x:.2f}, {y:.2f}) 失败: {e}")
```

### 2.2 测试标记点

在测试函数中还绘制了额外的测试标记：

```python
# 绘制测试参考点
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
        size=0.2,                     # 较大的测试点
        color=carla.Color(0, 255, 0), # 绿色测试点
        life_time=15.0
    )
    print(f"[INFO] 绘制测试点: {desc} at ({x:.2f}, {y:.2f}, {z:.2f})")
```

**绘制内容**：
- 🟢 **测试参考点**：绿色大点，用于验证坐标系统

---

## 3. 主控制逻辑：`agents/rule_based/rule_based_agent.py`

### 3.1 可视化调用逻辑

在主循环中的可视化调用：

```python
while True:
    # 1. 定期更新走廊并绘制边界
    if (planner.corridor is None) or (frame % planner.dp_interval == 0):
        planner.update_corridor(env.world, ego=env.ego)
        draw_corridor(env.world, ref, planner.corridor)  # 绘制走廊边界

    # 2. 推进仿真
    obs, _ = env.step()
    throttle, steer, brake, dbg = planner.compute_control(obs, dt=dt)

    # 3. 定期绘制自车位置标记
    if frame % 2 == 0:
        ego_pose = obs.get("ego_pose", {})
        draw_ego_marker(env.world, ego_pose.get("x", 0.0), ego_pose.get("y", 0.0))

    # 4. 应用控制
    env.apply_control(throttle=throttle, steer=steer, brake=brake)

    frame += 1
```

### 3.2 走廊更新中的障碍物可视化

```python
def update_corridor(self, world, ego=None, debug_draw_points: bool = True):
    # ... 障碍物检测逻辑 ...

    # 可选的障碍物点可视化
    if debug_draw_points and pts_se:
        draw_pts_se(world, self.ref, pts_se,
                   color=(0, 255, 0),    # 绿色障碍点
                   size=0.08,
                   life=0.8)
```

---

## 4. 可视化元素汇总

### 4.1 颜色编码系统

| 颜色 | RGB值 | 用途 | 含义 |
|------|-------|------|------|
| 🟢 绿色 | (0,255,0) | 走廊边界 | 安全通行区域 |
| 🟡 黄色 | (255,255,0) | 走廊中线 | 推荐行驶路径 |
| 🔵 蓝色 | (0,150,255) | 自车标记 | 车辆当前位置 |
| 🔴 红色 | (255,0,0) | 障碍物点 | 检测到的障碍物 |
| 🟢 亮绿 | (0,255,0) | Frenet障碍点 | Frenet坐标系障碍物 |

### 4.2 点大小设置

| 大小值 | 用途 | 可视效果 |
|--------|------|----------|
| 0.08 | 障碍物点 | 小点，密集显示 |
| 0.1 | 自车标记 | 中等点，清晰可见 |
| 0.15 | 测试障碍点 | 较大点，调试用 |
| 0.2 | 测试参考点 | 大点，突出显示 |

### 4.3 生存时间设置

| 时长(秒) | 用途 | 更新频率 |
|----------|------|----------|
| 0.1 | 自车标记 | 高频更新 |
| 0.2 | 走廊边界 | 中频更新 |
| 0.6 | Frenet障碍点 | 低频更新 |
| 5.0 | 测试障碍点 | 长时间显示 |
| 15.0 | 测试参考点 | 超长显示 |

### 4.4 高度设置策略

| 抬高值(米) | 用途 | 原因 |
|-----------|------|------|
| 0.05 | 障碍物点 | 避免与路面重叠 |
| 0.10 | 走廊边界线 | 清晰显示边界 |
| 0.30 | 自车标记 | 突出显示位置 |

---

## 5. 可视化系统的技术特点

### 5.1 坐标转换处理

```python
# Frenet坐标 → 世界坐标
x, y = ref.se2xy(s, ey)

# 世界坐标 → 路面投影
wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=0.0), project_to_road=True)
z = (wp.transform.location.z if wp else 0.0) + height_offset
```

### 5.2 Z-fighting避免策略

通过不同的高度偏移避免图形重叠：
- 路面：z = road_height
- 障碍物点：z = road_height + 0.05
- 走廊边界：z = road_height + 0.10
- 自车标记：z = road_height + 0.30

### 5.3 性能优化

- **降采样绘制**：自车标记每2帧绘制一次
- **短生存周期**：大多数元素0.1-0.6秒自动消失
- **条件绘制**：可通过`debug_draw_points`参数控制是否显示

### 5.4 错误处理

```python
try:
    # 绘制操作
    world.debug.draw_point(...)
except Exception as e:
    print(f"[DEBUG] 绘制失败: {e}")
    # 继续执行，不中断主流程
```

---

## 6. 实际使用效果

### 6.1 完整的可视化场景

当系统运行时，你会在CARLA仿真器中看到：

1. **🟢🟡 动态走廊**：绿色边界线和黄色中线实时更新，显示车辆的安全通行区域
2. **🔵 自车轨迹**：蓝色点串成的轨迹，显示车辆的历史路径
3. **🔴 障碍物云**：红色点云，显示检测到的各种障碍物
4. **📝 调试信息**：可选的文字标注，显示调试数据

### 6.2 调试价值

- **路径规划可视化**：直观看到规划的安全走廊
- **障碍物检测验证**：确认检测算法是否正确识别障碍物
- **控制效果评估**：通过轨迹判断控制算法性能
- **坐标系统验证**：确保Frenet和世界坐标转换正确

这套可视化系统为自动驾驶算法的开发和调试提供了强有力的工具支持！