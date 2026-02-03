# 快速开始指南

## 🚀 快速测试三个新场景

### 1️⃣ 启动 CARLA
```bash
cd /path/to/CARLA
./CarlaUE4.sh
```

### 2️⃣ 进入工作目录
```bash
cd /home/ajifang/RL_selector/agents/rule_based
```

### 3️⃣ 运行场景测试

#### 方法A: 单独测试每个场景
```bash
# 测试鬼探头场景（行人突然横穿）
python rule_based_agent.py --scenario jaywalker

# 测试 Trimma 场景（左右夹击 + 前车）
python rule_based_agent.py --scenario trimma

# 测试施工变道场景
python rule_based_agent.py --scenario construction

# 测试原有锥桶场景（对比基准）
python rule_based_agent.py --scenario cones
```

#### 方法B: 使用自动化测试脚本
```bash
# 测试所有场景（依次运行）
./test_scenarios.sh

# 只测试新增的三个场景
./test_scenarios.sh jaywalker trimma construction

# 测试单个场景
./test_scenarios.sh jaywalker
```

---

## 📊 查看测试结果

测试完成后，日志会保存在以下目录：

```
agents/rule_based/
├── logs_rule_based_jaywalker/      # 鬼探头场景日志
│   ├── telemetry.csv               # 遥测数据
│   ├── speed.png                   # 速度曲线
│   ├── controls.png                # 控制量曲线
│   └── ey_vs_s.png                 # 横向偏差图
├── logs_rule_based_trimma/         # Trimma场景日志
└── logs_rule_based_construction/   # 施工场景日志
```

---

## 🎯 场景简介

### Jaywalker（鬼探头）⭐⭐⭐⭐⭐
- **挑战**: 行人突然横穿马路
- **测试能力**: 紧急制动、避让反应
- **特点**: 高密度交通流 + 动态行人

### Trimma（左右夹击）⭐⭐⭐⭐
- **挑战**: 被前车和左右车包围
- **测试能力**: 超车决策、变道能力
- **特点**: 三车道 + 速度差异 + 交通流

### Construction（施工变道）⭐⭐⭐⭐
- **挑战**: 前方施工封道，必须变道
- **测试能力**: 找gap、安全变道
- **特点**: 施工区障碍物 + 高密度交通流

---

## 🔧 常见问题

### Q1: 场景初始化失败？
**A**: 重启 CARLA 服务器，确保使用 Town01-Town07 地图

### Q2: Ego 车辆生成失败？
**A**: 清理 CARLA 中的其他车辆，或重启服务器

### Q3: 控制频繁失败？
**A**: 降低参考速度（修改 `v_ref_base` 参数）

### Q4: 交通流生成失败？
**A**: 检查 `/home/ajifang/DriveAdapter/tools/custom_eval.py` 是否存在

---

## 📝 实时可视化说明

运行时 CARLA 中会显示：
- **灰色线**: 参考线（车道中心）
- **紫色线**: 左边界
- **绿色线**: 右边界
- **黄色线**: DP规划路径

---

## 🎮 控制台输出示例

```
[CTRL] s=45.2 ey=0.12 | lo=-1.85 up=1.92 w=3.77 | v=8.45->12.00 | delta=0.023 steer=0.02 | dp_ok=True opt_ok=True
[LONG] th=0.65 br=0.00
```

- `s`: 纵向距离
- `ey`: 横向偏差
- `w`: 走廊宽度
- `v`: 当前速度 -> 参考速度
- `th/br`: 油门/刹车

---

## 📚 详细文档

更多信息请查看：
- **完整文档**: `README_SCENARIOS.md`
- **场景源码**: `/home/ajifang/RL_selector/env/scenarios.py`
- **Planner源码**: `rule_based_agent.py`

---

## ✅ 验证清单

- [ ] CARLA 服务器已启动
- [ ] Python 环境已激活
- [ ] 依赖包已安装（carla, numpy, scipy）
- [ ] 场景文件存在（`env/scenarios.py`）
- [ ] 测试脚本可执行（`chmod +x test_scenarios.sh`）

---

## 🎉 开始测试！

```bash
# 一键测试所有场景
./test_scenarios.sh

# 或单独测试
python rule_based_agent.py --scenario jaywalker
```

**按 Ctrl+C 停止当前测试**

祝测试顺利！🚗💨
