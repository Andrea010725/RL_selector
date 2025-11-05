import os
import time
import random
from typing import List, Union, Optional, Dict, Tuple, Type
from dataclasses import dataclass

import carla
import ipdb
from tensorboard.compat.tensorflow_stub.tensor_shape import vector

from rl import CARLACollectWrapper, utils
from rl.environments.carla import env_utils as carla_utils
from rl import *
from core import CARLAEnv, CARLAgent


@dataclass
class AgentConfig:
    """Configuration class for CARLA agent parameters"""
    batch_size: int
    consider_obs_every: int
    skip_data: int
    load: bool
    class_: Type = CARLAgent


@dataclass
class EnvConfig:
    """Configuration class for CARLA environment parameters"""
    image_shape: Tuple[int, int, int]
    window_size: Tuple[int, int]
    town: Optional[str]
    render: bool
    debug: bool
    class_: Type = CARLAEnv


@dataclass
class StageConfig:
    """Base configuration for training stages"""
    episodes: int
    timesteps: int
    batch_size: int
    save_every: Optional[int]
    seed: int = 42
    stage_name: str = 'stage'
    load_model: bool = False
    town: Optional[str] = None
    traffic: Dict[str, int] = None


def sample_origins(amount: int = 1, seed: Optional[int] = None) -> Union[List[carla.Transform], carla.Transform]:
    """Sample spawn points for vehicles in CARLA environment.

    This function connects to CARLA simulator and samples available spawn points
    for vehicles. It can return either a single spawn point or multiple spawn points
    based on the amount parameter.

    Args:
        amount: Number of spawn points to sample. Defaults to 1.
        seed: Random seed for reproducibility. Defaults to None.

    Returns:
        If amount > 1, returns a list of carla.Transform objects representing spawn points.
        If amount = 1, returns a single carla.Transform object.

    Raises:
        AssertionError: If amount is less than or equal to 0.
        ConnectionError: If unable to connect to CARLA server.
    """
    # Validate input parameters
    assert amount > 0, "Amount must be positive"

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    try:
        # Connect to CARLA server and get world map
        client = carla_utils.get_client(
            address='localhost',
            port=2000,
            timeout=10.0
        )
        world_map = client.get_world().get_map()

        # Get all available spawn points
        available_points = world_map.get_spawn_points()

        if not available_points:
            raise ValueError("No spawn points available in the current map")

        # Return multiple spawn points if requested
        if amount > 1:
            random.shuffle(available_points)
            return available_points[:amount]

        # Return single spawn point
        return random.choice(available_points)

    except Exception as e:
        raise ConnectionError(f"Failed to connect to CARLA server: {str(e)}")


def define_agent(
        class_: Type = CARLAgent,
        batch_size: int = 128,
        consider_obs_every: int = 4,
        load: bool = False,
        **kwargs
) -> Dict:
    """Define configuration for CARLA reinforcement learning agent.

    Args:
        class_: Agent class to be used. Defaults to CARLAgent.
        batch_size: Size of training batches. Defaults to 128.
        consider_obs_every: Frequency of observation consideration. Defaults to 4.
        load: Whether to load pre-trained model. Defaults to False.
        **kwargs: Additional agent configuration parameters.

    Returns:
        Dict: Complete agent configuration dictionary.

    Example:
        >>> agent_config = define_agent(batch_size=64, load=True)
    """
    config = AgentConfig(
        class_=class_,
        batch_size=batch_size,
        consider_obs_every=consider_obs_every,
        skip_data=1,
        load=load
    )

    # Convert dataclass to dict and update with additional parameters
    agent_dict = {**config.__dict__, **kwargs}
    return agent_dict


def define_env(
        image_shape: Tuple[int, int, int] = (90, 120, 3),
        window_size: Tuple[int, int] = (1080, 270),
        town: Optional[str] = 'Town03',
        render: bool = True,
        debug: bool = False,
        **kwargs
) -> Dict:
    """Define configuration for CARLA environment.

    This function creates a configuration dictionary for the CARLA environment,
    specifying parameters such as image dimensions, window size, and simulation settings.

    Args:
        image_shape: Dimensions of input images (height, width, channels).
                    Defaults to (90, 120, 3).
        window_size: Size of rendering window (width, height).
                    Defaults to (1080, 270).
        town: CARLA town map to use. Defaults to 'Town03'.
        render: Whether to render the environment. Defaults to True.
        debug: Whether to enable debug mode. Defaults to False.
        **kwargs: Additional environment configuration parameters.

    Returns:
        Dict: Complete environment configuration dictionary.

    Example:
        >>> env_config = define_env(render=False, town='Town01')
    """
    config = EnvConfig(
        class_=CARLAEnv,
        image_shape=image_shape,
        window_size=window_size,
        town=town,
        render=render,
        debug=debug
    )

    # Convert dataclass to dict and update with additional parameters
    env_dict = {**config.__dict__, **kwargs}
    return env_dict


# -------------------------------------------------------------------------------------------------
# -- Stage
# -------------------------------------------------------------------------------------------------

class Stage:
    """Base class for reinforcement learning training stages"""

    def __init__(self, agent: dict, environment: dict, learning: dict, representation: dict = None,
                 collect: dict = None, name='Stage'):
        assert isinstance(agent, dict)
        assert isinstance(environment, dict)
        assert isinstance(learning, dict)

        # Initialize agent
        self.agent_class = agent.pop('class', agent.pop('class_'))
        self.agent_args = agent
        self.agent = None

        assert isinstance(learning['agent'], dict)

        # Initialize environment
        self.env_class = environment.pop('class', environment.pop('class_'))
        self.env_args = environment
        self.env = None

        # Representation learning configuration
        if isinstance(representation, dict):
            self.should_do_repr_lear = True
            self.repr_args = representation
        else:
            self.should_do_repr_lear = False

        # Data collection configuration
        if isinstance(collect, dict):
            self.should_collect = True
            self.collect_args = collect
            assert isinstance(learning['collect'], dict)
        else:
            self.should_collect = False

        self.learn_args = learning
        self.name = name

    def init(self):
        """Initialize environment and agent"""
        if self.env is None:
            self.env = self.env_class(**self.env_args)
            self.agent = self.agent_class(self.env, **self.agent_args)

    def run_train(self, epochs: int, copy_weights=True, epoch_offset=0) -> 'Stage':  # epoch_offset=0
        """Execute reinforcement learning training

        Args:
            epochs: Number of training epochs
            copy_weights: Whether to save weights
            epoch_offset: Epoch offset for weight saving

        Returns:
            self: Support for method chaining
        """
        assert epochs > 0
        self.init()

        #         for epoch in range(epochs):
        #             t0 = time.time()
        #             self.reinforcement_learning()
        #             print(f'[{self.name}] Epoch {epoch + 1}/{epochs} took {round(time.time() - t0, 3)}s.')

        #             if copy_weights:
        #                 utils.copy_folder(src=self.agent.base_path,
        #                                 dst=f'{self.agent.base_path}-{epoch + epoch_offset}')

        for epoch in range(epoch_offset, epoch_offset + epochs):  # 从40开始训练
            t0 = time.time()
            self.reinforcement_learning()
            print(
                f'[{self.name}] Epoch {epoch + 1 - epoch_offset}/{epochs} (Global: {epoch + 1}/100) took {round(time.time() - t0, 3)}s.')

            if copy_weights:
                utils.copy_folder(src=self.agent.base_path,
                                  dst=f'{self.agent.base_path}-{epoch}')  # 直接使用epoch存储
        self.cleanup()
        return self

    def evaluate(self, **kwargs) -> 'Stage':
        """Evaluate model performance"""
        self.init()
        self.agent.evaluate(**kwargs)
        return self

    def reinforcement_learning(self):
        """Execute reinforcement learning"""
        self.agent.learn(**self.learn_args['agent'])

    def cleanup(self):
        """Cleanup environment and agent"""
        self.env.close()
        self.env = None
        self.agent = None


# -------------------------------------------------------------------------------------------------
# -- Curriculum (stages definition)

def create_base_agent_config(config: StageConfig, **kwargs) -> Dict:
    """Create base agent configuration with common parameters"""
    base_config = {
        'class_': CARLAgent,
        'batch_size': config.batch_size,
        'name': config.stage_name,
        'seed': config.seed,
        'advantage_scale': 2.0,
        'load': config.load_model,
        'load_full': config.load_model,
        'traces_dir': None,
        'shuffle_batches': False,
        'drop_batch_remainder': True,
        'shuffle': True,
        'consider_obs_every': 1,
        'clip_norm': 1.0,
        'update_dynamics': True
    }

    base_config.update(kwargs)
    return define_agent(**base_config)


def create_base_env_config(config: StageConfig, **kwargs) -> Dict:
    """Create base environment configuration with common parameters"""
    base_config = {
        'town': config.town,
        'debug': True,
        'image_shape': (90, 120, 3),
        'throttle_as_desired_speed': True,
        'info_every': kwargs.pop('repeat_action', 1),
        'disable_reverse': True,
        'window_size': (900, 245)
    }

    base_config.update(kwargs)
    return define_env(**base_config)


def create_learning_params(config: StageConfig) -> Dict:
    """Create learning parameters configuration"""
    return {
        'agent': {
            'episodes': config.episodes,
            'timesteps': config.timesteps,
            'render_every': False,
            'close': False,
            'save_every': config.save_every
        }
    }


def stage_s1(episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42, stage_name='stage-s1', **kwargs):
    config = StageConfig(
        episodes=episodes,
        timesteps=timesteps,
        batch_size=batch_size,
        save_every=save_every,
        seed=seed,
        stage_name=stage_name,
        load_model=kwargs.pop('load', False)
    )

    env_kwargs = {'path': {'origin': sample_origins(amount=10, seed=seed)}}
    return Stage(
        agent=create_base_agent_config(config, **kwargs),
        environment=create_base_env_config(config, **env_kwargs),
        learning=create_learning_params(config)
    )


def stage_s2(episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42, stage_name='stage-s2', **kwargs):
    config = StageConfig(
        episodes=episodes,
        timesteps=timesteps,
        batch_size=batch_size,
        save_every=save_every,
        seed=seed,
        stage_name=stage_name,
        load_model=kwargs.pop('load', True)
    )

    env_kwargs = {
        'path': {'origin': sample_origins(amount=50, seed=seed)},
        'spawn': {'vehicles': 0, 'pedestrians': 50}
    }

    return Stage(
        agent=create_base_agent_config(config, **kwargs),
        environment=create_base_env_config(config, **env_kwargs),
        learning=create_learning_params(config)
    )


def stage_s3(episodes: int, timesteps: int, batch_size: int, save_every=None, seed=42, stage_name='stage-s3', **kwargs):
    config = StageConfig(
        episodes=episodes,
        timesteps=timesteps,
        batch_size=batch_size,
        save_every=save_every,
        seed=seed,
        stage_name=stage_name,
        load_model=kwargs.pop('load', True),
        traffic={'vehicles': 50, 'pedestrians': 50}
    )

    env_kwargs = {'spawn': config.traffic}

    return Stage(
        agent=create_base_agent_config(config, **kwargs),
        environment=create_base_env_config(config, **env_kwargs),
        learning=create_learning_params(config)
    )


def stage_s4(episodes: int, timesteps: int, batch_size: int, towns=None, save_every=None,
             seed=42, stage_name='stage-s4', **kwargs):
    config = StageConfig(
        episodes=episodes,
        timesteps=timesteps,
        batch_size=batch_size,
        save_every=save_every,
        seed=seed,
        stage_name=stage_name,
        load_model=kwargs.pop('load', True),
        traffic={'vehicles': 50, 'pedestrians': 50}
    )

    env_kwargs = {
        'random_towns': towns,
        'spawn': config.traffic
    }

    return Stage(
        agent=create_base_agent_config(config, **kwargs),
        environment=create_base_env_config(config, **env_kwargs),
        learning=create_learning_params(config)
    )


def stage_s5(episodes: int, timesteps: int, batch_size: int, towns='Town03', save_every=None,
             seed=42, stage_name='stage-s5', traffic='dense', **kwargs):
    traffic_spec = {
        'no': None,
        'regular': {'vehicles': 0, 'pedestrians': 0},
        'dense': {'vehicles': 0, 'pedestrians': 0}
    }

    config = StageConfig(
        episodes=episodes,
        timesteps=timesteps,
        batch_size=batch_size,
        save_every=save_every,
        seed=seed,
        stage_name=stage_name,
        load_model=kwargs.pop('load', True),
        town=towns,
        traffic=traffic_spec[traffic]
    )

    env_kwargs = {'spawn': config.traffic}

    return Stage(
        agent=create_base_agent_config(config, **kwargs),
        environment=create_base_env_config(config, **env_kwargs),
        learning=create_learning_params(config)
    )


# -------------------------------------------------------------------------------------------------
# -- EVALUATION
# -------------------------------------------------------------------------------------------------

def evaluate(mode: str, town: str, seeds: list, traffic: str, steps=512, trials=50, weights='stage-s5'):
    def make_stage():
        return stage_s5(
            episodes=1,
            timesteps=steps,
            batch_size=1,
            town=town,
            seed_regularization=True,
            stage_name=weights,
            aug_intensity=0.0,
            repeat_action=1,
            traffic=traffic,
            log_mode=None
        )

    stage = make_stage()

    for i, seed in enumerate(seeds):
        stage.evaluate(
            name=f'{weights}-{mode}-{steps}-{trials}-{town}-{traffic}-{seed}',
            timesteps=steps,
            trials=trials,
            town='Town03',
            seeds='sample',
            initial_seed=seed,
            close=i + 1 == len(seeds)
        )
