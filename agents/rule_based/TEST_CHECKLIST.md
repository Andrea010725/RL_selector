# 测试检查清单

## 📋 测试前准备

### 环境检查
- [ ] CARLA 服务器已启动（端口 2000）
- [ ] Python 环境已激活
- [ ] 依赖包已安装：
  - [ ] carla (0.9.15)
  - [ ] numpy
  - [ ] scipy
  - [ ] pandas
  - [ ] matplotlib

### 文件检查
```bash
cd /home/ajifang/RL_selector/agents/rule_based
python verify_scenarios.py
```
- [ ] 所有导入验证通过
- [ ] 所有文件完整性检查通过

---

## 🧪 场景测试清单

### 1. Cones（锥桶场景）- 基准测试

**测试命令**:
```bash
python rule_based_agent_0203.py --scenario cones
```

**预期行为**:
- [ ] 场景初始化成功
- [ ] Ego 在锥桶后方生成
- [ ] 锥桶单侧布置
- [ ] 走廊边界正确绘制（紫色/绿色线）
- [ ] DP 路径正确规划（黄色线）
- [ ] 车辆平稳绕行锥桶
- [ ] 无碰撞
- [ ] 控制成功率 > 95%

**关键指标**:
- [ ] 平均速度: 8-12 m/s
- [ ] 平均横向偏差: < 0.3 m
- [ ] 最大横向偏差: < 1.0 m

**日志检查**:
```bash
cd logs_rule_based_cones/
ls -lh  # 应该有 telemetry.csv, speed.png, controls.png, ey_vs_s.png
```

---

### 2. Jaywalker（鬼探头场景）

**测试命令**:
```bash
python rule_based_agent_0203.py --scenario jaywalker
```

**预期行为**:
- [ ] 场景初始化成功
- [ ] Ego 在直道上生成
- [ ] 行人在前方车道边缘生成
- [ ] 交通流生成成功（对向和相邻车道）
- [ ] 当 ego 接近时，行人开始横穿
- [ ] 行人移动流畅（无跳跃）
- [ ] Ego 能够检测到行人并减速/避让
- [ ] 无碰撞（行人和其他车辆）

**关键指标**:
- [ ] 行人触发成功
- [ ] 最小距离 > 2.0 m（与行人）
- [ ] 紧急制动响应时间 < 1.0 s
- [ ] 控制成功率 > 90%

**特殊检查**:
- [ ] 控制台输出 "[Jaywalker] ✅ 行人开始横穿"
- [ ] 行人从一侧移动到另一侧
- [ ] 行人到达后停止

**日志检查**:
```bash
cd logs_rule_based_jaywalker/
ls -lh
```

---

### 3. Trimma（左右夹击场景）

**测试命令**:
```bash
python rule_based_agent_0203.py --scenario trimma
```

**预期行为**:
- [ ] 场景初始化成功
- [ ] Ego 在中间车道生成
- [ ] 前车、左车、右车正确生成
- [ ] 三辆车速度差异明显（前车快，左右车慢）
- [ ] 交通流生成成功
- [ ] Ego 能够找到 gap 进行变道/超车
- [ ] 无碰撞

**关键指标**:
- [ ] 前车距离: ~18 m
- [ ] 左右车偏移: ~3 m
- [ ] 速度差异明显
- [ ] 成功超车或变道
- [ ] 控制成功率 > 85%

**特殊检查**:
- [ ] 控制台输出 "[Trimma] ✅ 场景生成成功"
- [ ] 打印前车、左车、右车位置
- [ ] 打印速度差百分比

**日志检查**:
```bash
cd logs_rule_based_trimma/
ls -lh
```

---

### 4. Construction（施工变道场景）

**测试命令**:
```bash
python rule_based_agent_0203.py --scenario construction
```

**预期行为**:
- [ ] 场景初始化成功
- [ ] Ego 在直道上生成
- [ ] 施工区在前方 30m 处生成
- [ ] 施工区包含锥桶、水马、垃圾、施工人员
- [ ] 相邻车道交通流生成成功
- [ ] Ego 能够检测到施工区
- [ ] Ego 能够找到 gap 变道
- [ ] 成功绕过施工区
- [ ] 无碰撞

**关键指标**:
- [ ] 施工区距离: ~30 m
- [ ] 交通密度: 3 辆/100m
- [ ] 成功变道
- [ ] 控制成功率 > 85%

**特殊检查**:
- [ ] 控制台输出 "[ConstructionLaneChange] ✅ 场景生成成功"
- [ ] 打印施工区位置
- [ ] 打印交通流车辆数量
- [ ] 打印相邻车道（左或右）

**调试检查**:
```bash
# 查看施工区 actors
# 控制台应该输出:
# [DEBUG] static.prop count = ...
# [DEBUG] walkers count = ...
```

**日志检查**:
```bash
cd logs_rule_based_construction/
ls -lh
```

---

## 📊 性能对比测试

### 批量测试（推荐）

**测试命令**:
```bash
python batch_test_scenarios.py --duration 60 --scenarios jaywalker trimma construction
```

**预期输出**:
```
场景对比测试报告
─────────────────────────────────────────
场景: 鬼探头场景（行人横穿）
  ✅ 测试成功
  总帧数: 1200
  平均速度: 8.45 m/s
  控制成功率: 92.3%
  ...
```

**对比指标**:
- [ ] 所有场景测试成功
- [ ] 生成对比报告
- [ ] 性能指标合理

---

## 🐛 故障排查清单

### 问题1: CARLA 连接失败
**症状**: `Connection refused` 或 `Timeout`

**检查步骤**:
- [ ] CARLA 服务器是否运行？
  ```bash
  nc -z localhost 2000 && echo "CARLA 运行中" || echo "CARLA 未运行"
  ```
- [ ] 端口是否被占用？
  ```bash
  lsof -i :2000
  ```
- [ ] 防火墙是否阻止？

**解决方案**:
```bash
# 重启 CARLA
cd /path/to/CARLA
./CarlaUE4.sh
```

---

### 问题2: 场景初始化失败
**症状**: `场景初始化失败` 或 `找不到合适道路`

**检查步骤**:
- [ ] 当前地图是否支持？（推荐 Town01-Town07）
- [ ] 地图是否有足够的多车道道路？
- [ ] 是否有其他车辆占用？

**解决方案**:
```bash
# 切换地图
# 在 CARLA 中按 M 键切换地图
# 或重启 CARLA 并加载指定地图
cd /path/to/CARLA
./CarlaUE4.sh -quality-level=Low Town03
```

---

### 问题3: Ego 生成失败
**症状**: `无法在场景提供的位置生成 ego`

**检查步骤**:
- [ ] 生成位置是否被占用？
- [ ] 生成位置是否在有效道路上？
- [ ] z 坐标是否合理？

**解决方案**:
```bash
# 清理所有车辆（在 CARLA Python API 中）
python -c "
import carla
client = carla.Client('localhost', 2000)
world = client.get_world()
for actor in world.get_actors().filter('vehicle.*'):
    actor.destroy()
print('已清理所有车辆')
"
```

---

### 问题4: 交通流生成失败
**症状**: `交通流生成失败` 或 `TrafficFlowSpawner not found`

**检查步骤**:
- [ ] `/home/ajifang/DriveAdapter/tools/custom_eval.py` 是否存在？
- [ ] Traffic Manager 端口是否被占用？
- [ ] 交通密度是否过高？

**解决方案**:
```python
# 方案1: 禁用交通流
# 在 main() 函数中设置:
config = SimpleNamespace(
    ...
    enable_traffic_flow=False,  # 禁用交通流
)

# 方案2: 降低交通密度
config = SimpleNamespace(
    ...
    traffic_density=1.0,  # 降低到 1 辆/100m
)
```

---

### 问题5: 控制频繁失败
**症状**: 频繁出现 `[CTRL-FAIL]`

**检查步骤**:
- [ ] 走廊宽度是否过窄？（查看 `w=...`）
- [ ] 参考速度是否过高？（查看 `v_ref=...`）
- [ ] DP 规划是否成功？（查看 `dp_ok=...`）

**解决方案**:
```python
# 在 RuleBasedPlanner 中调整参数:
planner = RuleBasedPlanner(amap, v_ref_base=8.0)  # 降低参考速度

# 或调整走廊参数:
planner.CONE_EXTRA_CLEAR = 1.2  # 增大安全距离（默认 0.9）
planner.DP_CORRIDOR_MARGIN = 0.30  # 增大走廊余量（默认 0.25）
```

---

### 问题6: 行人不移动（Jaywalker）
**症状**: 行人生成但不横穿

**检查步骤**:
- [ ] 是否触发？（查看控制台输出）
- [ ] 触发距离是否合理？
- [ ] `tick_update()` 是否被调用？

**解决方案**:
```python
# 检查 main() 函数中是否有:
if scenario_type == "jaywalker" and scenario is not None:
    ego_loc = ego.get_location()
    scenario.check_and_trigger(ego_loc)  # 检查触发
    scenario.tick_update()               # 更新行人位置
```

---

### 问题7: 日志文件缺失
**症状**: 日志目录为空或缺少文件

**检查步骤**:
- [ ] 是否运行足够长时间？（至少 10 秒）
- [ ] 是否正常退出？（Ctrl+C）
- [ ] TelemetryLogger 是否初始化？

**解决方案**:
```bash
# 检查日志目录
ls -lah logs_rule_based_*/

# 如果为空，检查是否有权限问题
chmod -R u+w logs_rule_based_*/
```

---

## ✅ 测试完成标准

### 基本标准（必须满足）
- [ ] 所有场景能够成功初始化
- [ ] Ego 车辆能够成功生成
- [ ] 无崩溃或异常退出
- [ ] 生成完整的日志文件

### 性能标准（建议满足）
- [ ] 控制成功率 > 85%
- [ ] 无碰撞（或碰撞率 < 5%）
- [ ] 平均速度合理（6-12 m/s）
- [ ] 横向偏差合理（< 0.5 m）

### 功能标准（场景特定）
- [ ] **Jaywalker**: 行人能够触发并横穿
- [ ] **Trimma**: 能够识别并处理包围情况
- [ ] **Construction**: 能够检测施工区并变道

---

## 📝 测试报告模板

```markdown
# Rule-Based Planner 场景测试报告

**测试日期**: YYYY-MM-DD
**测试人员**: [你的名字]
**CARLA 版本**: 0.9.15
**地图**: Town03

## 测试环境
- 操作系统: Linux
- Python 版本: 3.7
- 依赖版本: [列出关键依赖]

## 测试结果

### 1. Cones（锥桶场景）
- 状态: ✅ 通过 / ❌ 失败
- 平均速度: X.XX m/s
- 控制成功率: XX.X%
- 备注: [任何观察或问题]

### 2. Jaywalker（鬼探头场景）
- 状态: ✅ 通过 / ❌ 失败
- 行人触发: ✅ 成功 / ❌ 失败
- 平均速度: X.XX m/s
- 控制成功率: XX.X%
- 备注: [任何观察或问题]

### 3. Trimma（左右夹击场景）
- 状态: ✅ 通过 / ❌ 失败
- 超车/变道: ✅ 成功 / ❌ 失败
- 平均速度: X.XX m/s
- 控制成功率: XX.X%
- 备注: [任何观察或问题]

### 4. Construction（施工变道场景）
- 状态: ✅ 通过 / ❌ 失败
- 变道: ✅ 成功 / ❌ 失败
- 平均速度: X.XX m/s
- 控制成功率: XX.X%
- 备注: [任何观察或问题]

## 总结
- 通过场景数: X/4
- 总体评价: [优秀/良好/需改进]
- 主要问题: [列出主要问题]
- 改进建议: [列出改进建议]
```

---

## 🎯 下一步

测试完成后：

1. **分析日志**
   ```bash
   cd logs_rule_based_jaywalker/
   python -c "import pandas as pd; df=pd.read_csv('telemetry.csv'); print(df.describe())"
   ```

2. **对比性能**
   ```bash
   python batch_test_scenarios.py --duration 120 --scenarios cones jaywalker trimma construction
   ```

3. **调优参数**
   - 根据测试结果调整 planner 参数
   - 调整场景难度参数
   - 优化控制策略

4. **生成报告**
   - 填写测试报告模板
   - 保存日志和图表
   - 记录问题和改进建议

---

**祝测试顺利！** 🚗💨
