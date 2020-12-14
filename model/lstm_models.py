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

from config import config

# policy_kwargs = dict(n_lstm=64, net_arch=[
#                      64, 'lstm', dict(vf=[32, 32], pi=[32, 32])])

policy_kwargs = dict(n_lstm=128, net_arch=[
                     64, 64, 'lstm', dict(vf=[64, 32], pi=[64, 32])])


def train_lstm_A2C(env_train, model_name, model=None, timesteps=25000, save_path=None):
    """A2C model"""

    start = time.time()
    if model is None:
        model = A2C('MlpLnLstmPolicy', env_train,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=config.TENSORBOARD_DIR,
                    verbose=config.VERBOSE)
    else:
        model.set_env(env_train)
        model.verbose = config.VERBOSE

    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    end = time.time()

    model.save(save_path or f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print('Training time (A2C): ', (end - start) / 60, ' minutes')
    return model


def train_lstm_PPO(env_train, model_name, model=None, timesteps=50000, save_path=None):
    """PPO model"""

    start = time.time()
    if model is None:
        model = PPO2('MlpLnLstmPolicy', env_train, ent_coef=0.005, nminibatches=1,
                     policy_kwargs=policy_kwargs,
                     tensorboard_log=config.TENSORBOARD_DIR,
                     verbose=config.VERBOSE)
    else:
        model.set_env(env_train)
        model.verbose = config.VERBOSE

    model.learn(total_timesteps=timesteps, tb_log_name=model_name)
    end = time.time()

    if save_path is None:
        save_path = f"{config.TRAINED_MODEL_DIR}/{model_name}"
    model.save(save_path)
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    return model
