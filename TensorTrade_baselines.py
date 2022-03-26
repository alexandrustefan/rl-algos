#!/usr/bin/env python
# coding: utf-8

# # Install needed deps

# #### Don't forget to run ```apt-get update --fix-missing && sudo apt-get install build-essential``` and ```apt-get install zlib1g-dev``` in case you are running on an Ubuntu image

# In[1]:


get_ipython().system('pip install git+https://github.com/tensortrade-org/tensortrade.git')
get_ipython().system('pip install pandas-ta==0.3.14b --pre')
get_ipython().system('pip install gym==0.18.0')
get_ipython().system('pip install ipywidgets')
get_ipython().system('pip install stable-baselines3[extra]')


# # Prepare and fetch the data

# In[211]:


from tensortrade.data.cdd import CryptoDataDownload

import numpy as np
import pandas as pd
pd.options.mode.use_inf_as_na = True

def prepare_data(df):
    df['volume'] = np.int64(df['volume'])
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d %I:%M %p')
    return df

def fetch_data():
    cdd = CryptoDataDownload()
    exchange_data = cdd.fetch("Bitfinex", "BTC", "ETH", "1h")
    exchange_data = exchange_data[['date', 'open', 'high', 'low', 'close', 'volume']]
    exchange_data = prepare_data(exchange_data)
    return exchange_data

def load_csv(filename):
    df = pd.read_csv('data/' + filename, skiprows=1)
    df.drop(columns=['symbol', 'volume_btc'], inplace=True)

    # Fix timestamp from "2019-10-17 09-AM" to "2019-10-17 09-00-00 AM"
    df['date'] = df['date'].str[:14] + '00-00 ' + df['date'].str[-2:]

    return prepare_data(df)


# In[212]:


data = fetch_data()
data


# ## Create features for the feed module

# In[217]:


import os
import numpy as np
import ta as ta1
import pandas_ta as ta

import quantstats as qs
qs.extend_pandas()

def fix_dataset_inconsistencies(dataframe, fill_value=None):
    dataframe = dataframe.replace([-np.inf, np.inf], np.nan)

    #This is done to avoid filling middle holes with backfilling.
    if fill_value is None:
        dataframe.iloc[0,:] =             dataframe.apply(lambda column: column.iloc[column.first_valid_index()], axis='index')
    else:
        dataframe.iloc[0,:] =             dataframe.iloc[0,:].fillna(fill_value)

    return dataframe.fillna(axis='index', method='pad').dropna(axis='columns')

def rsi(price: 'pd.Series[pd.Float64Dtype]', period: float) -> 'pd.Series[pd.Float64Dtype]':
    r = price.diff()
    upside = np.minimum(r, 0).abs()
    downside = np.maximum(r, 0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100*(1 - (1 + rs) ** -1)

def macd(price: 'pd.Series[pd.Float64Dtype]', fast: float, slow: float, signal: float) -> 'pd.Series[pd.Float64Dtype]':
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal

def generate_all_default_quantstats_features(data):
    excluded_indicators = [
        'compare',
        'greeks',
        'information_ratio',
        'omega',
        'r2',
        'r_squared',
        'rolling_greeks',
        'warn',
    ]
    
    indicators_list = [f for f in dir(qs.stats) if f[0] != '_' and f not in excluded_indicators]
    
    df = data.copy()
    df = df.set_index('date')
    df.index = pd.DatetimeIndex(df.index)

    for indicator_name in indicators_list:
        try:
            #print(indicator_name)
            indicator = qs.stats.__dict__[indicator_name](df['close'])
            if isinstance(indicator, pd.Series):
                indicator = indicator.to_frame(name=indicator_name)
                df = pd.concat([df, indicator], axis='columns')
        except (pd.errors.InvalidIndexError, ValueError):
            pass

    df = df.reset_index()
    return df

def generate_features(data):

    # Generate all default indicators from ta library
    ta1.add_all_ta_features(data, 
                            'open', 
                            'high', 
                            'low', 
                            'close', 
                            'volume', 
                            fillna=True)

    # Naming convention across most technical indicator libraries
    data = data.rename(columns={'open': 'Open', 
                                'high': 'High', 
                                'low': 'Low', 
                                'close': 'Close', 
                                'volume': 'Volume'})
    data = data.set_index('date')

    # Custom indicators
    features = pd.DataFrame.from_dict({
        'prev_open': data['Open'].shift(1),
        'prev_high': data['High'].shift(1),
        'prev_low': data['Low'].shift(1),
        'prev_close': data['Close'].shift(1),
        'prev_volume': data['Volume'].shift(1),
        'vol_5': data['Close'].rolling(window=5).std().abs(),
        'vol_10': data['Close'].rolling(window=10).std().abs(),
        'vol_20': data['Close'].rolling(window=20).std().abs(),
        'vol_30': data['Close'].rolling(window=30).std().abs(),
        'vol_50': data['Close'].rolling(window=50).std().abs(),
        'vol_60': data['Close'].rolling(window=60).std().abs(),
        'vol_100': data['Close'].rolling(window=100).std().abs(),
        'vol_200': data['Close'].rolling(window=200).std().abs(),
        'ma_5': data['Close'].rolling(window=5).mean(),
        'ma_10': data['Close'].rolling(window=10).mean(),
        'ma_20': data['Close'].rolling(window=20).mean(),
        'ma_30': data['Close'].rolling(window=30).mean(),
        'ma_50': data['Close'].rolling(window=50).mean(),
        'ma_60': data['Close'].rolling(window=60).mean(),
        'ma_100': data['Close'].rolling(window=100).mean(),
        'ma_200': data['Close'].rolling(window=200).mean(),
        'ema_5': ta1.trend.ema_indicator(data['Close'], window=5, fillna=True),
        'ema_9': ta1.trend.ema_indicator(data['Close'], window=9, fillna=True),
        'ema_21': ta1.trend.ema_indicator(data['Close'], window=21, fillna=True),
        'ema_60': ta1.trend.ema_indicator(data['Close'], window=60, fillna=True),
        'ema_64': ta1.trend.ema_indicator(data['Close'], window=64, fillna=True),
        'ema_120': ta1.trend.ema_indicator(data['Close'], window=120, fillna=True),
        'lr_open': np.log(data['Open']).diff().fillna(0),
        'lr_high': np.log(data['High']).diff().fillna(0),
        'lr_low': np.log(data['Low']).diff().fillna(0),
        'lr_close': np.log(data['Close']).diff().fillna(0),
        'r_volume': data['Close'].diff().fillna(0),
        'rsi_5': rsi(data['Close'], period=5),
        'rsi_10': rsi(data['Close'], period=10),
        'rsi_100': rsi(data['Close'], period=100),
        'rsi_7': rsi(data['Close'], period=7),
        'rsi_28': rsi(data['Close'], period=28),
        'rsi_6': rsi(data['Close'], period=6),
        'rsi_14': rsi(data['Close'], period=14),
        'rsi_26': rsi(data['Close'], period=24),
        'macd_normal': macd(data['Close'], fast=12, slow=26, signal=9),
        'macd_short': macd(data['Close'], fast=10, slow=50, signal=5),
        'macd_long': macd(data['Close'], fast=200, slow=100, signal=50),
        'macd_wolfpack': macd(data['Close'], fast=3, slow=8, signal=9),
    })

    # Concatenate both manually and automatically generated features
    data = pd.concat([data, features], axis='columns').fillna(method='pad')

    # Remove potential column duplicates
    data = data.loc[:,~data.columns.duplicated()]

    # Revert naming convention
    data = data.rename(columns={'Open': 'open', 
                                'High': 'high', 
                                'Low': 'low', 
                                'Close': 'close', 
                                'Volume': 'volume'})

    data = data.reset_index()

    # Generate all default quantstats features
    df_quantstats = generate_all_default_quantstats_features(data)

    # Concatenate both manually and automatically generated features
    data = pd.concat([data, df_quantstats], axis='columns').fillna(method='pad')

    # Remove potential column duplicates
    data = data.loc[:,~data.columns.duplicated()]

    # A lot of indicators generate NaNs at the beginning of DataFrames, so remove them
    data = data.iloc[200:]
    data = data.reset_index(drop=True)

    data = fix_dataset_inconsistencies(data, fill_value=None)
    return data


# In[216]:


data = generate_features(data)
# remove not needed features
to_drop = ['others_dlr', 'compsum']
data = data.drop(columns=to_drop)
data.shape


# ## Remove features with low variance before splitting the dataset

# In[218]:


from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
date = data[['date']].copy()
data = data.drop(columns=['date'])
sel.fit(data)
data[data.columns[sel.get_support(indices=True)]]
data = pd.concat([date, data], axis='columns')
data


# # Setup which data to use for training and which data to use for evaluation of RL Model

# In[219]:


from sklearn.model_selection import train_test_split

def split_data(data):
    X = data.copy()
    y = X['close'].pct_change()

    X_train_test, X_valid, y_train_test, y_valid =         train_test_split(data, data['close'].pct_change(), train_size=0.67, test_size=0.33, shuffle=False)

    X_train, X_test, y_train, y_test =         train_test_split(X_train_test, y_train_test, train_size=0.50, test_size=0.50, shuffle=False)

    return X_train, X_test, X_valid, y_train, y_test, y_valid


# In[220]:


X_train, X_test, X_valid, y_train, y_test, y_valid =     split_data(data)


# ## Implement basic feature engineering

# In[154]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from feature_engine.selection import SelectBySingleFeaturePerformance


# In[225]:


from scipy.stats import iqr


def estimate_outliers(data):
    return iqr(data) * 1.5

def estimate_percent_gains(data, column='close'):
    returns = get_returns(data, column=column)
    gains = estimate_outliers(returns)
    return gains

def get_returns(data, column='close'):
    return fix_dataset_inconsistencies(data[[column]].pct_change(), fill_value=0)

def precalculate_ground_truths(data, column='close', threshold=None):
    returns = get_returns(data, column=column)
    gains = estimate_outliers(returns) if threshold is None else threshold
    binary_gains = (returns[column] > gains).astype(int)
    return binary_gains

def is_null(data):
    return data.isnull().sum().sum() > 0



rf = RandomForestClassifier(n_estimators=100, 
                            random_state=1990, 
                            n_jobs=8)

sel = SelectBySingleFeaturePerformance(variables=None, 
                                       estimator=rf, 
                                       scoring="roc_auc", 
                                       cv=5, 
                                       threshold=0.65)

sel.fit(X_train, precalculate_ground_truths(X_train, column='close'))


# In[226]:


feature_performance = pd.Series(sel.feature_performance_).sort_values(ascending=False)
feature_performance


# In[223]:


import matplotlib.pyplot as plt

feature_performance.plot.bar(figsize=(40, 10))
plt.title('Performance of ML models trained with individual features')
plt.ylabel('roc-auc')


# In[227]:


features_to_drop = sel.features_to_drop_
features_to_drop


# In[228]:


to_drop = list(set(features_to_drop) - set(['open', 'high', 'low', 'close', 'volume']))
len(to_drop)


# In[229]:


X_train = X_train.drop(columns=to_drop)
X_test = X_test.drop(columns=to_drop)
X_valid = X_valid.drop(columns=to_drop)

X_train.shape, X_test.shape, X_valid.shape


# In[230]:


X_train.columns.tolist()


# ## Normalize the dataset subsets to make the model converge faster

# In[231]:


from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

scaler_type = MinMaxScaler

def get_feature_scalers(X, scaler_type=scaler_type):
    scalers = []
    for name in list(X.columns[X.columns != 'date']):
        scalers.append(scaler_type().fit(X[name].values.reshape(-1, 1)))
    return scalers

def get_scaler_transforms(X, scalers):
    X_scaled = []
    for name, scaler in zip(list(X.columns[X.columns != 'date']), scalers):
        X_scaled.append(scaler.transform(X[name].values.reshape(-1, 1)))
    X_scaled = pd.concat([pd.DataFrame(column, columns=[name]) for name, column in                           zip(list(X.columns[X.columns != 'date']), X_scaled)], axis='columns')
    return X_scaled

def normalize_data(X_train, X_test, X_valid):
    X_train_test = pd.concat([X_train, X_test], axis='index')
    X_train_test_valid = pd.concat([X_train_test, X_valid], axis='index')

    X_train_test_dates = X_train_test[['date']]
    X_train_test_valid_dates = X_train_test_valid[['date']]

    X_train_test = X_train_test.drop(columns=['date'])
    X_train_test_valid = X_train_test_valid.drop(columns=['date'])

    train_test_scalers =         get_feature_scalers(X_train_test, 
                            scaler_type=scaler_type)
    train_test_valid_scalers =         get_feature_scalers(X_train_test_valid, 
                            scaler_type=scaler_type)

    X_train_test_scaled =         get_scaler_transforms(X_train_test, 
                              train_test_scalers)
    X_train_test_valid_scaled =         get_scaler_transforms(X_train_test_valid, 
                              train_test_scalers)
    X_train_test_valid_scaled_leaking =         get_scaler_transforms(X_train_test_valid, 
                              train_test_valid_scalers)

    X_train_test_scaled =         pd.concat([X_train_test_dates, 
                   X_train_test_scaled], 
                  axis='columns')
    X_train_test_valid_scaled =         pd.concat([X_train_test_valid_dates, 
                   X_train_test_valid_scaled], 
                  axis='columns')
    X_train_test_valid_scaled_leaking =         pd.concat([X_train_test_valid_dates, 
                   X_train_test_valid_scaled_leaking], 
                  axis='columns')

    X_train_scaled = X_train_test_scaled.iloc[:X_train.shape[0]]
    X_test_scaled = X_train_test_scaled.iloc[X_train.shape[0]:]
    X_valid_scaled = X_train_test_valid_scaled.iloc[X_train_test.shape[0]:]
    X_valid_scaled_leaking = X_train_test_valid_scaled_leaking.iloc[X_train_test.shape[0]:]

    return (train_test_scalers, 
            train_test_valid_scalers, 
            X_train_scaled, 
            X_test_scaled, 
            X_valid_scaled, 
            X_valid_scaled_leaking)


# In[232]:


train_test_scalers, train_test_valid_scalers, X_train_scaled, X_test_scaled, X_valid_scaled, X_valid_scaled_leaking =     normalize_data(X_train, X_test, X_valid)


# ## Save new feature set

# In[233]:


import os
cwd = os.getcwd()

train_csv = os.path.join(cwd, 'train.csv')
test_csv = os.path.join(cwd, 'test.csv')
valid_csv = os.path.join(cwd, 'valid.csv')
train_scaled_csv = os.path.join(cwd, 'train_scaled.csv')
test_scaled_csv = os.path.join(cwd, 'test_scaled.csv')
valid_scaled_csv = os.path.join(cwd, 'valid_scaled.csv')
valid_scaled_leaking_csv = os.path.join(cwd, 'valid_scaled_leaking.csv')


# # Write a renderer

# In[234]:


import matplotlib.pyplot as plt

from tensortrade.env.generic import Renderer


class PositionChangeChart(Renderer):
    def __init__(self, color: str = "orange"):
        self.color = "orange"

    def render(self, env, **kwargs):
        history = pd.DataFrame(env.observer.renderer_history)

        actions = list(history.action)
        price = list(history.close)

        buy = {}
        sell = {}

        for i in range(len(actions) - 1):
            a1 = actions[i]
            a2 = actions[i + 1]

            if a1 != a2:
                if a1 == 0 and a2 == 1:
                    buy[i] = price[i]
                else:
                    sell[i] = price[i] 

        buy = pd.Series(buy)
        sell = pd.Series(sell)

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        fig.suptitle("Performance")

        axs[0].plot(np.arange(len(price)), price, label="price", color=self.color)
        axs[0].scatter(buy.index, buy.values, marker="^", color="green")
        axs[0].scatter(sell.index, sell.values, marker="^", color="red")
        axs[0].set_title("Trading Chart")

        performance_df = pd.DataFrame().from_dict(env.action_scheme.portfolio.performance, orient='index')
        performance_df.plot(ax=axs[1])
        axs[1].set_title("Net Worth")

        plt.show()


# # Defining the environment

# In[235]:


import tensortrade.env.default as default

from tensortrade.feed.core import DataFeed, Stream
from tensortrade.feed.core.base import NameSpace
from tensortrade.env.default.actions import BSH
from tensortrade.env.default.rewards import PBR, RiskAdjustedReturns, SimpleProfit
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.instruments import USD, BTC, ETH
from tensortrade.oms.wallets import Wallet, Portfolio
from tensortrade.oms.orders import TradeType

def create_env(config):
    # TODO: adjust according to your commission percentage, if present
    commission = 0.075
    price = Stream.source(list(X_train["close"]), dtype="float").rename("BTC-ETH")
    bitstamp_options = ExchangeOptions(commission=commission)
    bitstamp = Exchange("bitstamp", 
                        service=execute_order, 
                        options=bitstamp_options)(price)

    cash = Wallet(bitstamp, 10000 * ETH)
    asset = Wallet(bitstamp, 0 * BTC)

    portfolio = Portfolio(BTC, [cash, asset])

    with NameSpace("bitstamp"):
        features = [
            Stream.source(list(X_train_scaled[c]), 
                          dtype="float").rename(c) for c in X_train_scaled.columns[1:]
        ]

    feed = DataFeed(features)
    feed.compile()

    #reward_scheme = PBR(price=price)
    reward_scheme = SimpleProfit(window_size=config['window_size'])

    action_scheme = BSH(
        cash=cash,
        asset=asset
    )

    renderer_feed = DataFeed([
        Stream.source(list(X_train["date"])).rename("date"),
        Stream.source(list(X_train["open"]), dtype="float").rename("open"),
        Stream.source(list(X_train["high"]), dtype="float").rename("high"),
        Stream.source(list(X_train["low"]), dtype="float").rename("low"),
        Stream.source(list(X_train["close"]), dtype="float").rename("close"), 
        Stream.source(list(X_train["volume"]), dtype="float").rename("volume"), 
            Stream.sensor(action_scheme, 
                          lambda s: s.action, dtype="float").rename("action")
    ])

    renderer = [
        PositionChangeChart(),
        default.renderers.PlotlyTradingChart(
            display=True,  # show the chart on screen (default)
            height=1200,  # affects both displayed and saved file height. None for 100% height.
            include_plotlyjs = True
            
        ),
    ]

    max_allowed_loss = config["max_allowed_loss"]
    min_periods = config["window_size"]

    observer = default.observers.TensorTradeObserver(
        portfolio=portfolio,
        feed=feed,
        renderer_feed=renderer_feed,
        window_size=config['window_size'],
        min_periods=min_periods
    )

    stopper = default.stoppers.MaxLossStopper(
        max_allowed_loss=max_allowed_loss
    )

    informer = default.informers.TensorTradeInformer()

    random_start_pct = 0.0

    env = default.create(
        portfolio=portfolio,
        action_scheme=action_scheme,
        reward_scheme=reward_scheme,
        feed=feed,
        renderer_feed=renderer_feed,
        renderer=renderer,
        observer=observer,
        stopper=stopper,
        informer=informer,
        min_periods=min_periods,
        random_start_pct=random_start_pct,
        window_size=config['window_size']
    )
    
    return env


# In[236]:


get_ipython().run_line_magic('xmode', 'Plain')
get_ipython().run_line_magic('pdb', 'on')


# # Create the training environment and the training model

# In[237]:


import os
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy


cwd = os.getcwd()
logdir = "logs"
if not os.path.exists(logdir):
    os.makedirs(logdir)
    
models_dir = "models/PPO_" + time.strftime("%Y%m%d_%H_%M_%S")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

env_config_training = {
    "window_size": 14,  # We want to look at the last 14 samples (hours)
    "reward_window_size": 7,  # And calculate reward based on the actions taken in the next 7 hours
    "max_allowed_loss": 0.10,  # If it goes past 10% loss during the iteration, we don't want to waste time on a "loser".
}

env = Monitor(create_env(env_config_training))

model = PPO(MlpPolicy, env, verbose=1, tensorboard_log=logdir)
env.observer.feed.next()


# # Run an evaluation test to demonstrate random agent vs trained agent

# In[186]:


# env_config_evaluation = {
#     "window_size": 14,  # We want to look at the last 14 samples (hours)
#     "reward_window_size": 7,  # And calculate reward based on the actions taken in the next 7 hours
#     "max_allowed_loss": 1.00,  # During validation runs we want to see how bad it would go. Even up to 100% loss.
#     "csv_filename": os.path.join(cwd, 'evaluation.csv'),  # The variable that will be used to differentiate training and validation datasets
# }
# Use a separate environement for evaluation
# eval_env = Monitor(create_env(env_config_evaluation))

# Random Agent, before training
# mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=14)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


# # Train a PPO model and save it after each x steps with stable lines

# In[238]:


# it will save a model at each 10k steps. this helps us to choose the best one
TIMESTEPS = 10000
iters = 0
while True:
    iters += 1
    
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")


# # Load saved model

# In[188]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[239]:


# Make sure to load the model that performed the best, you can check it up in the tensorboard
# usualy it's the one with the highest rollout/ep_rew_mean, you can identify it by looking at the step number
model_path = f"{models_dir}/20000.zip"
model = PPO.load(model_path, env=env)


# # Render results

# In[241]:



steps = 5000
render_interval = steps // 10
obs = env.reset()
for i in range(steps):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.env.step(action)
    # if (i % render_interval) == 0:
    #     env.env.render()
    # if done:
    #     obs = env.reset()


# In[242]:


env.env.render()


# In[62]:





# In[ ]:




