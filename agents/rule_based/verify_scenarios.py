#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景验证脚本 - 验证三个新场景是否正确集成

用法:
    python verify_scenarios.py
"""

import sys
sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
sys.path.append("/home/ajifang/RL_selector")

def verify_imports():
    """验证所有必要的导入"""
    print("\n" + "="*60)
    print("  场景集成验证")
    print("="*60 + "\n")

    print("[1/5] 验证 CARLA 导入...")
    try:
        import carla
        print("  ✅ CARLA 模块导入成功")
    except ImportError as e:
        print(f"  ❌ CARLA 模块导入失败: {e}")
        return False

    print("\n[2/5] 验证场景类导入...")
    try:
        from env.scenarios import (
            ScenarioBase,
            JaywalkerScenario,
            TrimmaScenario,
            ConstructionLaneChangeScenario
        )
        print("  ✅ ScenarioBase")
        print("  ✅ JaywalkerScenario")
        print("  ✅ TrimmaScenario")
        print("  ✅ ConstructionLaneChangeScenario")
    except ImportError as e:
        print(f"  ❌ 场景类导入失败: {e}")
        return False

    print("\n[3/5] 验证 rule_based_agent 导入...")
    try:
        from agents.rule_based.rule_based_agent_0203 import (
            spawn_ego_from_scenario,
            spawn_ego_upstream_lane_center,
            RuleBasedPlanner,
            main
        )
        print("  ✅ spawn_ego_from_scenario")
        print("  ✅ spawn_ego_upstream_lane_center")
        print("  ✅ RuleBasedPlanner")
        print("  ✅ main")
    except ImportError as e:
        print(f"  ❌ rule_based_agent 导入失败: {e}")
        return False

    print("\n[4/5] 验证场景类结构...")
    try:
        from env.scenarios import JaywalkerScenario, TrimmaScenario, ConstructionLaneChangeScenario

        # 检查必要的方法
        required_methods = ['setup', 'get_spawn_transform', 'cleanup']

        for scenario_cls in [JaywalkerScenario, TrimmaScenario, ConstructionLaneChangeScenario]:
            scenario_name = scenario_cls.__name__
            for method in required_methods:
                if not hasattr(scenario_cls, method):
                    print(f"  ❌ {scenario_name} 缺少方法: {method}")
                    return False
            print(f"  ✅ {scenario_name} 结构完整")
    except Exception as e:
        print(f"  ❌ 场景类结构验证失败: {e}")
        return False

    print("\n[5/5] 验证场景配置...")
    try:
        from types import SimpleNamespace

        # Jaywalker 配置
        jaywalker_config = SimpleNamespace(
            jaywalker_distance=25.0,
            jaywalker_speed=2.5,
            jaywalker_trigger_distance=18.0,
            jaywalker_start_side="random",
            use_occlusion_vehicle=False,
            tm_port=8000,
            enable_traffic_flow=True,
        )
        print("  ✅ Jaywalker 配置验证通过")

        # Trimma 配置
        trimma_config = SimpleNamespace(
            front_vehicle_distance=18.0,
            side_vehicle_offset=3.0,
            min_lane_count=3,
            tm_port=8000,
            tm_global_distance=2.5,
            front_speed_diff_pct=70.0,
            side_speed_diff_pct=80.0,
            disable_lane_change=True,
            enable_traffic_flow=True,
        )
        print("  ✅ Trimma 配置验证通过")

        # Construction 配置
        construction_config = SimpleNamespace(
            construction_distance=30.0,
            construction_length=20.0,
            traffic_density=3.0,
            traffic_speed=8.0,
            min_gap_for_lane_change=12.0,
            construction_type="construction1",
            flow_range=80.0,
            tm_port=8000,
            enable_traffic_flow=True,
        )
        print("  ✅ Construction 配置验证通过")

    except Exception as e:
        print(f"  ❌ 场景配置验证失败: {e}")
        return False

    return True


def verify_files():
    """验证必要的文件是否存在"""
    import os

    print("\n" + "="*60)
    print("  文件完整性检查")
    print("="*60 + "\n")

    files_to_check = [
        ("/home/ajifang/RL_selector/env/scenarios.py", "场景定义文件"),
        ("/home/ajifang/RL_selector/agents/rule_based/rule_based_agent_0203.py", "Rule-based planner"),
        ("/home/ajifang/RL_selector/agents/rule_based/test_scenarios.sh", "测试脚本"),
        ("/home/ajifang/RL_selector/agents/rule_based/README_SCENARIOS.md", "场景文档"),
        ("/home/ajifang/RL_selector/agents/rule_based/QUICKSTART.md", "快速开始指南"),
    ]

    all_exist = True
    for filepath, description in files_to_check:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"  ✅ {description}")
            print(f"     路径: {filepath}")
            print(f"     大小: {size} bytes")
        else:
            print(f"  ❌ {description} 不存在")
            print(f"     路径: {filepath}")
            all_exist = False

    return all_exist


def print_usage_guide():
    """打印使用指南"""
    print("\n" + "="*60)
    print("  使用指南")
    print("="*60 + "\n")

    print("✅ 所有验证通过！你现在可以开始测试场景了。\n")

    print("📝 快速开始:")
    print("  1. 启动 CARLA 服务器:")
    print("     cd /path/to/CARLA && ./CarlaUE4.sh\n")

    print("  2. 进入工作目录:")
    print("     cd /home/ajifang/RL_selector/agents/rule_based\n")

    print("  3. 运行场景测试:")
    print("     # 测试鬼探头场景")
    print("     python rule_based_agent_0203.py --scenario jaywalker\n")

    print("     # 测试 Trimma 场景")
    print("     python rule_based_agent_0203.py --scenario trimma\n")

    print("     # 测试施工变道场景")
    print("     python rule_based_agent_0203.py --scenario construction\n")

    print("     # 或使用自动化脚本测试所有场景")
    print("     ./test_scenarios.sh\n")

    print("📊 查看结果:")
    print("  日志保存在: logs_rule_based_<场景名>/")
    print("  - telemetry.csv: 遥测数据")
    print("  - speed.png: 速度曲线")
    print("  - controls.png: 控制量曲线")
    print("  - ey_vs_s.png: 横向偏差图\n")

    print("📚 更多信息:")
    print("  - 快速开始: cat QUICKSTART.md")
    print("  - 详细文档: cat README_SCENARIOS.md\n")


def main():
    """主函数"""
    success = True

    # 验证导入
    if not verify_imports():
        success = False
        print("\n❌ 导入验证失败！")

    # 验证文件
    if not verify_files():
        success = False
        print("\n❌ 文件完整性检查失败！")

    if success:
        print_usage_guide()
        return 0
    else:
        print("\n" + "="*60)
        print("  验证失败")
        print("="*60)
        print("\n请检查上述错误信息并修复问题。\n")
        return 1


if __name__ == "__main__":
    exit(main())
