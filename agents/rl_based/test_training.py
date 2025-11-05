#!/usr/bin/env python3
"""
Simple training script for testing CARLA RL environment
使用早期版本的简化测试脚本
"""

import sys
sys.path.append("/home/ajifang/czw/RL_selector/agents/rl_based/")

from core.learning import stage_s1

def main():
    """简单的训练测试"""
    print("Starting simple CARLA RL training test...")

    # 创建Stage 1配置，使用最小参数
    stage = stage_s1(
        episodes=5,  # 只训练5个episode来测试
        timesteps=64,  # 每个episode只跑64步
        batch_size=32,  # 小批次大小
        seed=42,
        stage_name='test-stage',
        load=False  # 不加载预训练模型
    )

    # 开始训练
    print("Initializing training stage...")
    stage.run_train(epochs=1)  # 只运行1个epoch

    print("Training test completed!")

if __name__ == '__main__':
    main()