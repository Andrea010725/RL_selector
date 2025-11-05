import os

# Disable TensorFlow and CUDA logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Disable pygame welcome message
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
# Disable ALSA warnings
os.environ['ALSA_CONFIG_DIR'] = '/dev/null'

import sys
import logging
from typing import List, Optional, Dict
import tensorflow as tf
import pygame
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import yaml
import argparse
import ipdb

from core import learning


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from yaml file

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_environment(config: Dict):
    """Setup environment configuration including GPU and logging

    Args:
        config: Configuration dictionary containing environment settings
    """
    # Disable all TensorFlow logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    tf.get_logger().setLevel(logging.ERROR)

    # Disable CUDA devices info
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Disable pygame welcome message
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

    # Disable ALSA lib warnings
    # Redirect ALSA errors to /dev/null
    os.environ['ALSA_CONFIG_DIR'] = '/dev/null'

    # Disable gym warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)

    # Setup GPU if needed
    if not config['gpu_settings']['use_gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Add CARLA egg to path
    try:
        sys.path.append('carla-0.9.10-py3.7-linux-x86_64.egg')
    except IndexError as e:
        print(f"Failed to setup CARLA environment: {e}")
        sys.exit(1)


class StageConfig:
    """Training configuration for different stages"""

    def __init__(self, config: Dict):
        """Initialize stage configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config

    def get_base_config(self) -> Dict:
        """Get base configuration shared across all stages

        Returns:
            Dictionary containing base configuration
        """
        return self.config['training']['base_config']

    def get_stage_config(self, stage_num: int) -> Dict:
        """Get stage-specific configuration

        Args:
            stage_num: Stage number (1-5)

        Returns:
            Dictionary containing merged base and stage-specific configuration
        """
        base_config = self.get_base_config()
        stage_config = self.config['training']['stage_configs'][str(stage_num)]
        return {**base_config, **stage_config}


def create_stage(config: Dict, stage_num: int, **additional_params):
    """Factory function to create stage instances

    Args:
        config: Configuration dictionary
        stage_num: Stage number (1-5)
        additional_params: Additional parameters to override configuration

    Returns:
        Stage instance
    """
    stage_config = StageConfig(config).get_stage_config(stage_num)
    stage_config.update(additional_params)

    stage_constructors = {
        1: learning.stage_s1,
        2: learning.stage_s2,
        3: learning.stage_s3,
        4: learning.stage_s4,
        5: learning.stage_s5
    }

    return stage_constructors[stage_num](**stage_config)


def train(config: Dict, stage_num: int, use_expert_pretrain: bool = False):
    """Train specified stage

    Args:
        config: Configuration dictionary
        stage_num: Stage number to train
        use_expert_pretrain: Whether to use expert data pre-training
    """
    # Create expert pretraining configs if enabled
    expert_config = None
    pretrain_config = None

    # if use_expert_pretrain:
    #     expert_config = ExpertDataConfig(
    #         data_dir=config.get('expert_data', {}).get('data_dir', 'expert_data'),
    #         batch_size=config.get('expert_data', {}).get('batch_size', 32),
    #         max_files=config.get('expert_data', {}).get('max_files', None)
    #     )
    #     pretrain_config = PretrainingConfig(
    #         learning_rate=config.get('pretraining', {}).get('learning_rate', 1e-4),
    #         epochs=config.get('pretraining', {}).get('epochs', 50),
    #         early_stopping_patience=config.get('pretraining', {}).get('patience', 10)
    #     )

    # 检查是否指定了特定的权重检查点
    additional_params = {}
    if 'stage_name' in config:
        additional_params['stage_name'] = config['stage_name']
        print(f"使用指定的权重检查点: {config['stage_name']}")

    stage = create_stage(config, stage_num, **additional_params)
    print(f"Training Stage {stage_num}")

    # 检查训练流程配置
    training_flow = config.get('training_flow', {})
    expert_pretrain_only = training_flow.get('expert_pretrain_only', False)
    skip_rl_training = training_flow.get('skip_rl_training', False)

    if expert_pretrain_only or skip_rl_training:
        print("配置为仅进行专家数据预训练，跳过RL训练")
        if use_expert_pretrain:
            # 仅进行专家数据预训练
            stage.run_expert_pretrain_only(
                expert_config=expert_config,
                pretrain_config=pretrain_config
            )
        else:
            print("警告：配置为专家数据预训练模式，但未启用--expert-pretrain参数")
    else:
        # 完整训练流程：专家数据预训练 + RL训练
        stage.run_train(
            epochs=config['training']['epochs'],
            epoch_offset=config['training']['epoch_offset'],
            # use_expert_pretrain=use_expert_pretrain,
            # expert_config=expert_config,
            # pretrain_config=pretrain_config
        )


def evaluate(config: Dict):
    """Evaluate model across different towns and traffic conditions

    Args:
        config: Configuration dictionary
    """
    eval_config = config['evaluation']
    for mode in ['test']:
        for town in eval_config['towns']:
            for traffic in eval_config['traffic_conditions']:
                print(f'Evaluation config: [mode={mode}, town={town}, traffic={traffic}, steps={eval_config["steps"]}]')
                learning.evaluate(
                    mode,
                    town=town,
                    steps=eval_config['steps'],
                    seeds=eval_config['seeds'],
                    traffic=traffic
                )


def main():
    """Main entry point of the program"""
    parser = argparse.ArgumentParser(description='CARLA Driving RL Agent')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True,
                        help='Run mode: train or evaluate')
    parser.add_argument('--stage', type=int, choices=range(1, 6),
                        help='Training stage (1-5)')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--expert-pretrain', action='store_true',
                        help='Enable expert data pre-training before RL training')

    args = parser.parse_args()
    config = load_config(args.config)
    setup_environment(config)

    try:
        if args.mode == 'train':
            if args.stage is None:
                raise ValueError("Training mode requires --stage parameter")
            train(config, args.stage, args.expert_pretrain)
        else:  # evaluate
            evaluate(config)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
