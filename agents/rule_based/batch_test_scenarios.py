#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
场景对比测试脚本 - 批量运行所有场景并生成对比报告

用法:
    python batch_test_scenarios.py [--duration 60] [--scenarios jaywalker trimma construction]
"""

import sys
import os
import time
import argparse
import subprocess
from datetime import datetime

sys.path.append("/home/ajifang/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg")
sys.path.append("/home/ajifang/RL_selector")


def check_carla_running():
    """检查 CARLA 服务器是否运行"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', 2000))
    sock.close()
    return result == 0


def run_scenario(scenario_name, duration=60):
    """
    运行单个场景测试

    Args:
        scenario_name: 场景名称
        duration: 运行时长（秒）

    Returns:
        (success, log_dir, error_msg)
    """
    print(f"\n{'='*60}")
    print(f"  开始测试场景: {scenario_name}")
    print(f"  预计运行时长: {duration}秒")
    print(f"{'='*60}\n")

    log_dir = f"logs_rule_based_{scenario_name}"

    # 构建命令
    cmd = [
        "timeout", str(duration),
        "python", "rule_based_agent_0203.py",
        "--scenario", scenario_name
    ]

    start_time = time.time()

    try:
        # 运行测试
        result = subprocess.run(
            cmd,
            cwd="/home/ajifang/RL_selector/agents/rule_based",
            capture_output=True,
            text=True,
            timeout=duration + 10  # 额外10秒缓冲
        )

        elapsed = time.time() - start_time

        # timeout 命令返回 124 表示超时（正常）
        if result.returncode == 124 or result.returncode == 0:
            print(f"\n✅ 场景 {scenario_name} 测试完成")
            print(f"   运行时长: {elapsed:.1f}秒")
            print(f"   日志目录: {log_dir}/")
            return True, log_dir, None
        else:
            error_msg = result.stderr if result.stderr else "未知错误"
            print(f"\n❌ 场景 {scenario_name} 测试失败")
            print(f"   错误信息: {error_msg}")
            return False, log_dir, error_msg

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\n✅ 场景 {scenario_name} 测试完成（超时正常退出）")
        print(f"   运行时长: {elapsed:.1f}秒")
        print(f"   日志目录: {log_dir}/")
        return True, log_dir, None

    except Exception as e:
        print(f"\n❌ 场景 {scenario_name} 测试异常")
        print(f"   异常信息: {str(e)}")
        return False, log_dir, str(e)


def analyze_logs(log_dir):
    """
    分析日志文件，提取关键指标

    Returns:
        dict: 包含各项指标的字典
    """
    import pandas as pd
    import numpy as np

    csv_path = os.path.join(log_dir, "telemetry.csv")

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)

        metrics = {
            "total_frames": len(df),
            "avg_speed": df["v"].mean() if "v" in df.columns else 0,
            "max_speed": df["v"].max() if "v" in df.columns else 0,
            "avg_throttle": df["throttle"].mean() if "throttle" in df.columns else 0,
            "avg_brake": df["brake"].mean() if "brake" in df.columns else 0,
            "avg_abs_ey": df["ey"].abs().mean() if "ey" in df.columns else 0,
            "max_abs_ey": df["ey"].abs().max() if "ey" in df.columns else 0,
            "control_success_rate": (df["opt_ok"].sum() / len(df) * 100) if "opt_ok" in df.columns else 0,
        }

        return metrics

    except Exception as e:
        print(f"   ⚠️ 日志分析失败: {e}")
        return None


def generate_report(results):
    """
    生成测试报告

    Args:
        results: 测试结果列表
    """
    print("\n" + "="*80)
    print("  场景对比测试报告")
    print("="*80 + "\n")

    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 场景描述
    scenario_desc = {
        "cones": "锥桶场景（基准）",
        "jaywalker": "鬼探头场景（行人横穿）",
        "trimma": "Trimma场景（左右夹击）",
        "construction": "施工变道场景"
    }

    # 打印每个场景的结果
    for result in results:
        scenario = result["scenario"]
        success = result["success"]
        metrics = result["metrics"]

        print(f"{'─'*80}")
        print(f"场景: {scenario_desc.get(scenario, scenario)}")
        print(f"{'─'*80}")

        if not success:
            print(f"  ❌ 测试失败")
            print(f"  错误: {result.get('error', '未知错误')}\n")
            continue

        if metrics is None:
            print(f"  ⚠️ 无法分析日志\n")
            continue

        print(f"  ✅ 测试成功")
        print(f"  总帧数: {metrics['total_frames']}")
        print(f"  平均速度: {metrics['avg_speed']:.2f} m/s")
        print(f"  最大速度: {metrics['max_speed']:.2f} m/s")
        print(f"  平均油门: {metrics['avg_throttle']:.3f}")
        print(f"  平均刹车: {metrics['avg_brake']:.3f}")
        print(f"  平均横向偏差: {metrics['avg_abs_ey']:.3f} m")
        print(f"  最大横向偏差: {metrics['max_abs_ey']:.3f} m")
        print(f"  控制成功率: {metrics['control_success_rate']:.1f}%")
        print(f"  日志目录: {result['log_dir']}/")
        print()

    # 对比表格
    print("="*80)
    print("  性能对比")
    print("="*80 + "\n")

    # 表头
    print(f"{'场景':<20} {'平均速度':<12} {'横向偏差':<12} {'控制成功率':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    for result in results:
        if result["success"] and result["metrics"]:
            scenario = result["scenario"]
            metrics = result["metrics"]
            print(f"{scenario:<20} {metrics['avg_speed']:>8.2f} m/s {metrics['avg_abs_ey']:>8.3f} m {metrics['control_success_rate']:>9.1f}%")

    print()


def main():
    parser = argparse.ArgumentParser(description="批量测试多个场景并生成对比报告")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="每个场景的运行时长（秒），默认60秒"
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["jaywalker", "trimma", "construction"],
        choices=["cones", "jaywalker", "trimma", "construction"],
        help="要测试的场景列表"
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("  Rule-Based Planner 批量场景测试")
    print("="*80)
    print(f"\n测试场景: {', '.join(args.scenarios)}")
    print(f"每场景时长: {args.duration}秒")
    print(f"预计总时长: {args.duration * len(args.scenarios)}秒 ({args.duration * len(args.scenarios) / 60:.1f}分钟)\n")

    # 检查 CARLA
    print("[检查] CARLA 服务器状态...")
    if not check_carla_running():
        print("❌ CARLA 服务器未运行！")
        print("\n请先启动 CARLA:")
        print("  cd /path/to/CARLA")
        print("  ./CarlaUE4.sh\n")
        return 1
    print("✅ CARLA 服务器正在运行\n")

    # 运行测试
    results = []

    for i, scenario in enumerate(args.scenarios, 1):
        print(f"\n[{i}/{len(args.scenarios)}] 测试场景: {scenario}")

        success, log_dir, error = run_scenario(scenario, args.duration)

        # 分析日志
        metrics = None
        if success:
            print(f"\n[分析] 正在分析日志...")
            metrics = analyze_logs(log_dir)
            if metrics:
                print(f"   ✅ 日志分析完成")
            else:
                print(f"   ⚠️ 日志分析失败或日志不存在")

        results.append({
            "scenario": scenario,
            "success": success,
            "log_dir": log_dir,
            "error": error,
            "metrics": metrics
        })

        # 等待一下，让 CARLA 稳定
        if i < len(args.scenarios):
            print(f"\n[等待] 准备下一个场景...")
            time.sleep(5)

    # 生成报告
    generate_report(results)

    # 统计
    success_count = sum(1 for r in results if r["success"])
    print("="*80)
    print(f"  测试完成: {success_count}/{len(results)} 个场景成功")
    print("="*80 + "\n")

    return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    exit(main())
