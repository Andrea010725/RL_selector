# Rule-Based Planner 多场景测试文档

## 概述

本项目已集成了多个测试场景，用于评估 Rule-Based Planner 在不同驾驶场景下的性能。

## 支持的场景

### 1. Cones（锥桶场景）- 原有场景
**场景描述**: 原有的锥桶绕行场景，测试车辆在锥桶约束下的路径规划能力。

**特点**:
- 锥桶单侧布置
- 需要在锥桶和车道边界之间找到安全走廊
- 测试基础的避障和路径跟踪能力

**运行命令**:
```bash
python rule_based_agent.py --scenario cones
# 或
./test_scenarios.sh cones
```

---

### 2. Jaywalker（鬼探头场景）
**场景描述**: 行人突然从道路一侧横穿到另一侧，测试紧急制动和避让能力。

**特点**:
- 行人在自车前方一定距离处横穿马路
- 当自车接近到触发距离时，行人开始移动
- 包含高密度交通流（对向车道和相邻车道）
- 考验紧急制动能力和反应速度

**配置参数**:
- `jaywalker_distance`: 25.0m（行人位置距离自车spawn点）
- `jaywalker_speed`: 2.5 m/s（行人移动速度）
- `jaywalker_trigger_distance`: 18.0m（触发距离）
- `enable_traffic_flow`: True（启用交通流）

**运行命令**:
```bash
python rule_based_agent.py --scenario jaywalker
# 或
./test_scenarios.sh jaywalker
```

**难度**: ⭐⭐⭐⭐⭐ 非常困难

---

### 3. Trimma（左右夹击场景）
**场景描述**: 自车被其他车辆包围（前车、左车、右车），需要找到合适的gap进行超车或变道。

**特点**:
- 前车速度较慢（慢70%）
- 左右车速度更慢（慢80%）
- 三车道结构（左、中、右）
- 包含额外的高密度交通流
- 考验变道决策和超车能力

**配置参数**:
- `front_vehicle_distance`: 18.0m（前车距离）
- `side_vehicle_offset`: 3.0m（左右车相对自车的纵向偏移）
- `front_speed_diff_pct`: 70.0%（前车慢70%）
- `side_speed_diff_pct`: 80.0%（左右车慢80%）
- `enable_traffic_flow`: True（启用交通流）

**运行命令**:
```bash
python rule_based_agent.py --scenario trimma
# 或
./test_scenarios.sh trimma
```

**难度**: ⭐⭐⭐⭐ 困难

---

### 4. Construction（施工变道场景）
**场景描述**: 自车所在车道前方有施工封道区域（锥桶/水马/杂物/施工人员），必须向相邻车道变道绕行。

**特点**:
- 施工区域包含锥桶、水马、垃圾、施工人员
- 相邻车道存在高密度交通流（gap小，不容易插入）
- 需要找到合适的gap进行变道
- 考验"找gap + 安全变道 + 避让施工区"的综合能力

**配置参数**:
- `construction_distance`: 30.0m（施工区距离自车）
- `construction_length`: 20.0m（施工区长度）
- `traffic_density`: 3.0 辆/100m（交通密度）
- `traffic_speed`: 8.0 m/s（交通流速度）
- `construction_type`: "construction1"（施工类型）
- `enable_traffic_flow`: True（启用交通流）

**运行命令**:
```bash
python rule_based_agent.py --scenario construction
# 或
./test_scenarios.sh construction
```

**难度**: ⭐⭐⭐⭐ 困难

---

## 使用方法

### 方法1: 直接运行单个场景
```bash
cd /home/ajifang/RL_selector/agents/rule_based
python rule_based_agent.py --scenario <场景名称>
```

场景名称可选:
- `cones`: 锥桶场景
- `jaywalker`: 鬼探头场景
- `trimma`: 左右夹击场景
- `construction`: 施工变道场景

### 方法2: 使用测试脚本
```bash
cd /home/ajifang/RL_selector/agents/rule_based

# 测试所有场景
./test_scenarios.sh

# 测试指定场景
./test_scenarios.sh jaywalker trimma

# 测试单个场景
./test_scenarios.sh construction
```

### 方法3: Python 代码调用
```python
from rule_based_agent import main

# 运行指定场景
main(scenario_type="jaywalker")
```

---

## 前置条件

1. **启动 CARLA 服务器**:
   ```bash
   cd /path/to/CARLA
   ./CarlaUE4.sh
   ```

2. **确保依赖已安装**:
   - carla (0.9.15)
   - numpy
   - scipy
   - matplotlib (用于日志可视化)

3. **确保路径正确**:
   - CARLA Python API 路径已添加到 sys.path
   - 场景文件 `env/scenarios.py` 存在
   - 工具文件 `env/tools.py` 存在

---

## 输出和日志

每个场景运行后会生成独立的日志目录:

```
agents/rule_based/
├── logs_rule_based_cones/          # 锥桶场景日志
│   ├── telemetry.csv
│   ├── speed.png
│   ├── controls.png
│   └── ey_vs_s.png
├── logs_rule_based_jaywalker/      # 鬼探头场景日志
├── logs_rule_based_trimma/         # Trimma场景日志
└── logs_rule_based_construction/   # 施工场景日志
```

日志内容包括:
- `telemetry.csv`: 遥测数据（速度、位置、控制量等）
- `speed.png`: 速度曲线图
- `controls.png`: 控制量曲线图（油门、刹车、转向）
- `ey_vs_s.png`: 横向偏差 vs 纵向距离图

---

## 场景对比

| 场景 | 主要挑战 | 难度 | 交通流 | 动态障碍物 |
|------|---------|------|--------|-----------|
| Cones | 静态避障 | ⭐⭐⭐ | 可选 | 无 |
| Jaywalker | 紧急制动 | ⭐⭐⭐⭐⭐ | 是 | 行人 |
| Trimma | 超车/变道 | ⭐⭐⭐⭐ | 是 | 前车+左右车 |
| Construction | 变道+找gap | ⭐⭐⭐⭐ | 是 | 施工区 |

---

## 调试和可视化

### 实时可视化
运行时会在 CARLA 中绘制:
- **灰色线**: 参考线（车道中心线）
- **紫色线**: 物理左边界
- **绿色线**: 物理右边界
- **黄色线**: DP规划的中心路径

### 控制台输出
每10帧打印一次控制信息:
```
[CTRL] s=45.2 ey=0.12 | lo=-1.85 up=1.92 w=3.77 | v=8.45->12.00 | delta=0.023 steer=0.02 | dp_ok=True opt_ok=True | msg=ok
[LONG] th=0.65 br=0.00
```

参数说明:
- `s`: 纵向距离（沿参考线）
- `ey`: 横向偏差
- `lo/up`: 走廊下界/上界
- `w`: 走廊宽度
- `v`: 当前速度
- `v_ref`: 参考速度
- `delta`: 前轮转角
- `steer`: 转向控制量
- `th/br`: 油门/刹车

---

## 故障排查

### 问题1: 场景初始化失败
**症状**: 提示 "场景初始化失败"

**解决方法**:
1. 检查 CARLA 地图是否支持（建议使用 Town01-Town07）
2. 检查是否有足够的道路空间（某些场景需要多车道）
3. 尝试重启 CARLA 服务器

### 问题2: Ego 车辆生成失败
**症状**: 提示 "无法在场景提供的位置生成 ego"

**解决方法**:
1. 检查场景生成的位置是否被占用
2. 尝试清理 CARLA 中的其他车辆
3. 重启 CARLA 服务器

### 问题3: 交通流生成失败
**症状**: 提示 "交通流生成失败"

**解决方法**:
1. 检查 `DriveAdapter/tools/custom_eval.py` 是否存在
2. 检查 Traffic Manager 端口是否被占用
3. 降低交通密度参数

### 问题4: 控制失败
**症状**: 频繁出现 "[CTRL-FAIL]"

**解决方法**:
1. 检查走廊是否过窄（调整 `CONE_EXTRA_CLEAR` 参数）
2. 检查参考线是否有效
3. 降低参考速度 `v_ref_base`

---

## 自定义场景参数

可以在 `rule_based_agent.py` 的 `main()` 函数中修改场景配置参数。

例如，修改 Jaywalker 场景的行人速度:
```python
config = SimpleNamespace(
    jaywalker_distance=25.0,
    jaywalker_speed=3.5,  # 改为 3.5 m/s（更快）
    jaywalker_trigger_distance=18.0,
    ...
)
```

---

## 性能评估指标

建议从以下维度评估 planner 性能:

1. **安全性**:
   - 是否发生碰撞
   - 与障碍物的最小距离
   - 是否超出车道边界

2. **舒适性**:
   - 加速度变化率（jerk）
   - 横向加速度
   - 转向角变化率

3. **效率**:
   - 平均速度
   - 完成时间
   - 路径长度

4. **鲁棒性**:
   - 在不同场景下的成功率
   - 控制失败的频率
   - 恢复能力

---

## 贡献和反馈

如需添加新场景或修改现有场景，请参考 `env/scenarios.py` 中的 `ScenarioBase` 基类实现。

所有场景必须实现:
- `setup()`: 场景初始化和障碍物生成
- `get_spawn_transform()`: 返回自车生成位置
- `cleanup()`: 清理场景中生成的所有actors

---

## 更新日志

**2026-02-02**:
- ✅ 集成 JaywalkerScenario（鬼探头场景）
- ✅ 集成 TrimmaScenario（左右夹击场景）
- ✅ 集成 ConstructionLaneChangeScenario（施工变道场景）
- ✅ 添加场景选择命令行参数
- ✅ 添加自动化测试脚本
- ✅ 完善日志和清理机制

---

## 联系方式

如有问题或建议，请联系项目维护者。
