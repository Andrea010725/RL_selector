#!/bin/bash
# Rule-Based Planner 多场景测试脚本

echo "============================================"
echo "  Rule-Based Planner 多场景测试"
echo "============================================"
echo ""

# 检查 CARLA 是否运行
echo "[检查] 检查 CARLA 服务器是否运行..."
if ! nc -z localhost 2000 2>/dev/null; then
    echo "[错误] CARLA 服务器未运行！请先启动 CARLA："
    echo "  cd /path/to/CARLA"
    echo "  ./CarlaUE4.sh"
    exit 1
fi
echo "[成功] CARLA 服务器正在运行"
echo ""

# 进入脚本所在目录
cd "$(dirname "$0")"

# 测试场景列表
scenarios=("cones" "jaywalker" "trimma" "construction")

# 如果提供了参数，只测试指定场景
if [ $# -gt 0 ]; then
    scenarios=("$@")
fi

echo "将测试以下场景: ${scenarios[*]}"
echo ""

# 逐个测试场景
for scenario in "${scenarios[@]}"; do
    echo "============================================"
    echo "  测试场景: $scenario"
    echo "============================================"

    case $scenario in
        cones)
            echo "场景描述: 原有锥桶场景"
            ;;
        jaywalker)
            echo "场景描述: 鬼探头场景（行人突然横穿）"
            ;;
        trimma)
            echo "场景描述: Trimma场景（左右夹击 + 前车）"
            ;;
        construction)
            echo "场景描述: 施工封道 + 高密度交通流变道场景"
            ;;
        *)
            echo "[错误] 未知场景: $scenario"
            continue
            ;;
    esac

    echo ""
    echo "[运行] python rule_based_agent.py --scenario $scenario"
    echo "[提示] 按 Ctrl+C 停止当前场景测试"
    echo ""

    # 运行测试
    python rule_based_agent.py --scenario "$scenario"

    exit_code=$?

    if [ $exit_code -eq 0 ] || [ $exit_code -eq 130 ]; then
        echo ""
        echo "[完成] 场景 $scenario 测试完成"
        echo "[日志] 日志保存在: logs_rule_based_$scenario/"
        echo ""
    else
        echo ""
        echo "[错误] 场景 $scenario 测试失败 (退出码: $exit_code)"
        echo ""
        read -p "是否继续测试下一个场景？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "[退出] 用户取消测试"
            exit 1
        fi
    fi

    # 等待用户确认继续
    if [ "$scenario" != "${scenarios[-1]}" ]; then
        echo ""
        read -p "按 Enter 继续测试下一个场景，或 Ctrl+C 退出..."
        echo ""
    fi
done

echo ""
echo "============================================"
echo "  所有场景测试完成！"
echo "============================================"
echo ""
echo "日志文件位置："
for scenario in "${scenarios[@]}"; do
    echo "  - logs_rule_based_$scenario/"
done
echo ""
