#!/bin/bash
# 一键运行所有验证和测试

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Rule-Based Planner 场景集成 - 完整验证和测试             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# 进入工作目录
cd "$(dirname "$0")"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 步骤1: 检查 CARLA
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 1/4: 检查 CARLA 服务器"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if nc -z localhost 2000 2>/dev/null; then
    echo -e "${GREEN}✅ CARLA 服务器正在运行${NC}"
else
    echo -e "${RED}❌ CARLA 服务器未运行！${NC}"
    echo ""
    echo "请先启动 CARLA:"
    echo "  cd /path/to/CARLA"
    echo "  ./CarlaUE4.sh"
    echo ""
    exit 1
fi

echo ""

# 步骤2: 运行验证脚本
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 2/4: 验证场景集成"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python verify_scenarios.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ 验证失败！请检查上述错误信息。${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ 验证通过！${NC}"
echo ""

# 步骤3: 询问是否运行测试
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 3/4: 场景测试"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "现在可以开始测试场景。请选择测试模式："
echo ""
echo "  1) 快速测试单个场景（推荐新手）"
echo "  2) 批量测试所有场景（自动化）"
echo "  3) 手动测试（自己选择场景）"
echo "  4) 跳过测试"
echo ""

read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "选择要测试的场景："
        echo "  1) Jaywalker（鬼探头）"
        echo "  2) Trimma（左右夹击）"
        echo "  3) Construction（施工变道）"
        echo "  4) Cones（锥桶基准）"
        echo ""
        read -p "请输入选项 (1-4): " scenario_choice

        case $scenario_choice in
            1) scenario="jaywalker" ;;
            2) scenario="trimma" ;;
            3) scenario="construction" ;;
            4) scenario="cones" ;;
            *)
                echo -e "${RED}无效选项${NC}"
                exit 1
                ;;
        esac

        echo ""
        echo -e "${BLUE}开始测试场景: $scenario${NC}"
        echo -e "${YELLOW}提示: 按 Ctrl+C 停止测试${NC}"
        echo ""
        sleep 2

        python rule_based_agent_0203.py --scenario "$scenario"
        ;;

    2)
        echo ""
        read -p "每个场景运行多少秒？(默认60): " duration
        duration=${duration:-60}

        echo ""
        echo -e "${BLUE}开始批量测试，每个场景 ${duration} 秒${NC}"
        echo ""
        sleep 2

        python batch_test_scenarios.py --duration "$duration" --scenarios jaywalker trimma construction
        ;;

    3)
        echo ""
        echo "手动测试模式"
        echo ""
        echo "使用以下命令测试场景："
        echo "  python rule_based_agent.py --scenario jaywalker"
        echo "  python rule_based_agent.py --scenario trimma"
        echo "  python rule_based_agent.py --scenario construction"
        echo "  python rule_based_agent.py --scenario cones"
        echo ""
        echo "或使用测试脚本:"
        echo "  ./test_scenarios.sh jaywalker"
        echo ""
        ;;

    4)
        echo ""
        echo -e "${YELLOW}跳过测试${NC}"
        ;;

    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

echo ""

# 步骤4: 显示文档
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 4/4: 查看文档"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "📚 可用文档:"
echo ""
echo "  1. QUICKSTART.md          - 快速开始指南"
echo "  2. README_SCENARIOS.md    - 完整场景文档"
echo "  3. INTEGRATION_SUMMARY.md - 集成总结"
echo "  4. TEST_CHECKLIST.md      - 测试检查清单"
echo ""

read -p "是否查看快速开始指南？(y/n) " view_doc

if [[ $view_doc =~ ^[Yy]$ ]]; then
    echo ""
    cat QUICKSTART.md | head -50
    echo ""
    echo "（完整内容请查看 QUICKSTART.md）"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  设置完成！                                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "📊 查看测试结果:"
echo "  cd logs_rule_based_<场景名>/"
echo "  ls -lh"
echo ""

echo "🔧 常用命令:"
echo "  # 测试单个场景"
echo "  python rule_based_agent.py --scenario jaywalker"
echo ""
echo "  # 批量测试"
echo "  python batch_test_scenarios.py --duration 60"
echo ""
echo "  # 查看文档"
echo "  cat QUICKSTART.md"
echo "  cat README_SCENARIOS.md"
echo ""

echo -e "${GREEN}祝测试顺利！🚗💨${NC}"
echo ""
