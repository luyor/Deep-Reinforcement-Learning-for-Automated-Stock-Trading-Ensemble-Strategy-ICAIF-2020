# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
from stable_baselines import GAIL, SAC
from stable_baselines import ACER
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import DDPG
from stable_baselines import TD3

from stable_baselines.ddpg.policies import DDPGPolicy
from stable_baselines.common.policies import MlpPolicy
from config import config


def train_A2C(env_train, model_name, timesteps=25000, save_path=None):
    """A2C model"""

    start = time.time()
    model = A2C('MlpPolicy', env_train,
                tensorboard_log=config.TENSORBOARD_DIR)
    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    end = time.time()

    model.save(save_path or f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_ACER(env_train, model_name, timesteps=25000):
    start = time.time()
    model = ACER('MlpPolicy', env_train, verbose=0)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_DDPG(env_train, model_name, timesteps=10000):
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    start = time.time()
    model = DDPG('MlpPolicy', env_train, param_noise=param_noise,
                 action_noise=action_noise,
                 tensorboard_log=config.TENSORBOARD_DIR)
    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (DDPG): ', (end-start)/60, ' minutes')
    return model


def train_TD3(env_train, model_name, timesteps=30000):
    """TD3 model"""
    # add the noise objects for TD3
    n_actions = env_train.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(
        n_actions), sigma=0.1 * np.ones(n_actions))

    start = time.time()
    model = TD3('MlpPolicy', env_train, action_noise=action_noise)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (TD3): ', (end-start)/60, ' minutes')
    return model


def train_PPO(env_train, model_name, timesteps=50000, save_path=None):
    """PPO model"""

    start = time.time()
    model = PPO2('MlpPolicy', env_train, ent_coef=0.005, nminibatches=8,
                 tensorboard_log=config.TENSORBOARD_DIR)
    # model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    end = time.time()

    if save_path is None:
        save_path = f"{config.TRAINED_MODEL_DIR}/{model_name}"
    model.save(save_path)
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model


def train_GAIL(env_train, model_name, timesteps=1000):
    """GAIL Model"""
    # from stable_baselines.gail import ExportDataset, generate_expert_traj
    start = time.time()
    # generate expert trajectories
    model = SAC('MLpPolicy', env_train, verbose=1)
    generate_expert_traj(model, 'expert_model_gail',
                         n_timesteps=100, n_episodes=10)

    # Load dataset
    dataset = ExpertDataset(
        expert_path='expert_model_gail.npz', traj_limitation=10, verbose=1)
    model = GAIL('MLpPolicy', env_train, dataset, verbose=1)

    model.learn(total_timesteps=1000)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model
