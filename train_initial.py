from model.mlp_models import train_A2C, train_PPO
from model.lstm_models import train_lstm_A2C, train_lstm_PPO
from stable_baselines.common.vec_env import DummyVecEnv
from env.EnvMultipleStock_train import StockEnvTrain
from env.EnvMultipleStock_trade import StockEnvTrade
from preprocessing.preprocessors import data_split
from stable_baselines import A2C, PPO2
from config import config

import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def train_initial_model(train_func, df, timesteps, model_name, save_path):
    print("============Start Training Initial Model============")
    train = data_split(df, start=20090000, end=20151001)
    env_train = DummyVecEnv([lambda: StockEnvTrain(train)])
    return train_func(env_train, model_name, timesteps=timesteps, save_path=save_path)


def test_model(df, model, model_name, turbulence_threshold=140, start=20151001, end=20200707):
    trade_data = data_split(
        df, start=start, end=end)
    env_trade = DummyVecEnv([lambda: StockEnvTrade(trade_data,
                                                   turbulence_threshold=turbulence_threshold,
                                                   initial=True,
                                                   previous_state=[],
                                                   model_name=model_name,
                                                   iteration=0)])

    obs_trade = env_trade.reset()

    state = None
    dones = [False for _ in range(env_trade.num_envs)]
    for i in range(len(trade_data.index.unique())):
        action, state = model.predict(obs_trade, state=state, mask=dones)
        obs_trade, rewards, dones, info = env_trade.step(action)
    return info[0]


def evaluate_model(df, model, model_name):
    sharpe_list = []
    asset_list = []
    runs = 5
    for _ in range(runs):
        info = test_model(df, model, model_name, start=20151001, end=20161001)
        sharpe_list.append(info["sharpe"])
        asset_list.append(info["end_total_asset"])
    print("=================================")
    print("evaluating model: ", model_name)
    print(f"mean of {runs} runs:")
    print("sharp ratio: ", np.mean(sharpe_list))
    print("end_total_asset: ", np.mean(asset_list))


def initial_train_test(data):
    # model_name = "A2C_300k_dow_lstm64"
    # model_name = "A2C_300k_dow_lstm128"
    # model_name = "A2C_300k_dow_mlp"

    # model_name = "PPO_300k_dow_lstm64"
    model_name = "PPO_300k_dow_lstm128"
    # model_name = "PPO_500k_dow_mlp"

    save_path = f"trained_models/{model_name}"

    # load model
    # model_name = "PPO_100k_dow_1071"
    # save_path = f"trained_models/lstm32/{model_name}"

    # model = train_initial_model(train_A2C, data, 300000, model_name, save_path)
    # model = train_initial_model(train_PPO, data, 100000, model_name, save_path)

    # model = train_initial_model(
    #     train_lstm_A2C, data, 300000, model_name, save_path)
    model = train_initial_model(
        train_lstm_PPO, data, 300000, model_name, save_path)

    # model = A2C.load(load_path=save_path)
    # model = PPO2.load(load_path=save_path)

    evaluate_model(data, model, model_name)


def run():
    # read and preprocess data
    preprocessed_path = "done_data.csv"
    if os.path.exists(preprocessed_path):
        data = pd.read_csv(preprocessed_path, index_col=0)
    else:
        data = preprocess_data()
        data = add_turbulence(data)
        data.to_csv(preprocessed_path)

    print(data.head())
    print(data.size)

    initial_train_test(data)


if __name__ == "__main__":
    run()
