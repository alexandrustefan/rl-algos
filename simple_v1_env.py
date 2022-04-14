import random
import gym
from gym import spaces
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# infinite number in python
MAX_NET_WORTH = 2147483647
MAX_NUM_QUOTE_OR_BASE_ASSET = 2147483647

INITIAL_QUOTE_ASSET = 10000
INITIAL_BASE_ASSET = 10000
OBSERVATION_WINDOW_SIZE = 24 # Probably we should put it as param ?

class SimpleTradingEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, df, trading_fee):
        
        self.df = df
        self.window_size = OBSERVATION_WINDOW_SIZE
        self.features = self._process_data(df)
        self.obs_shape = (OBSERVATION_WINDOW_SIZE, self.features.shape[1])

        # Action space
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3.0, 1.0]), dtype=np.float32)
        # Observation space
        self.observation_space = spaces.Box(low=-1, high = 1, shape=self.obs_shape, dtype=np.float32)

        # Initialize the episode environment

        self._start_candle = OBSERVATION_WINDOW_SIZE # We assume that the first observation is not the first row of the dataframe, in order to avoid the case where there are no calculated indicators.
        self._end_candle = len(self.features) - 1
        self._trading_fee = trading_fee

        self._quote_asset = None
        self._base_asset = None
        self._done = None
        self._current_candle = None
        self._net_worth = None
        self._previous_net_worth = None

        # Render and analysis data
        self._action_history = None
        self._total_reward_accumulated = None
        self.trade_history = None # will be a dict
        self._first_rendering = None
        self._last_action_type = None
        

    def reset(self):
        self._done = False
        self._current_candle = self._start_candle
        self._quote_asset = INITIAL_QUOTE_ASSET
        self._base_asset = INITIAL_BASE_ASSET 
        self._net_worth = INITIAL_QUOTE_ASSET # at the begining our net worth is the initial quote asset
        self._previous_net_worth = INITIAL_QUOTE_ASSET # at the begining our previous net worth is the initial quote asset
        self._action_history = []
        self._total_reward_accumulated = 0.
        self._first_rendering = True
        self._last_action_type = 'Buy'
        self.trade_history = {}
        return self._get_observation()

    def _take_action(self, action):
        self._done = False
        self._current_candle += 1
        current_price = random.uniform(
            self.df.loc[self._current_candle, "low"], self.df.loc[self._current_candle, "high"])

        if self._current_candle == self._end_candle:
            self._done = True

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy % assets
            # Determine the maximum amount of quote asset that can be bought
            available_amount_to_buy_with = int(self._quote_asset / current_price)
            # Buy only the amount that agent chose
            assets_bought = int(available_amount_to_buy_with * amount)
            # Update the quote asset
            self._base_asset += assets_bought
            # substract trading fee from base asset based on the amount bought
            self._base_asset -= self._trading_fee * assets_bought
        

        elif action_type < 2:
            # Sell % assets
            # Determine the amount of base asset that can be sold
            amount_to_sell = int(self._quote_asset * amount)
            received_quote_asset = amount_to_sell * current_price
            self._quote_asset += received_quote_asset
            
            # substract trading fee from quote asset based on the amount sold
            self._quote_asset -= self._trading_fee * received_quote_asset

        # Update the current net worth
        self._net_worth = self._base_asset * current_price + self._quote_asset

        # Add action to action history array as tuple
        self._action_history.append(self._generate_action_data_tuple(action, current_price))
        self._last_action_type = self._get_human_readable_action(action)



    def step(self, action):
        """
        Returns the next observation, reward, done and info.
        """
        self._current_candle += 1
        self._take_action(action)

        # Calculate reward comparing the current net worth with the previous net worth
        reward = self._net_worth - self._previous_net_worth

        self._total_reward_accumulated += reward

        # Update the previous net worth to be the current net worth after the reward has been applied
        self._previous_net_worth = self._net_worth

        obs = self._get_observation()
        # Update the info and add it to history data
        info = dict (
            total_reward_accumulated = self._total_reward_accumulated,
            net_worth = self._net_worth,
            action_type = self._get_human_readable_action(action),
            action_amount = action[1],
        )

        self._update_trade_history(info)


        return obs, reward, self._done, info


    def _get_observation(self):
        """
        Returns the current observation.
        """
        data_frame = self.features[(self._current_candle - self.window_size):self._current_candle]

        #obs = np.append(data_frame, [[self._net_worth, self._quote_asset, self._base_asset]], axis=0)

        return data_frame

    def _update_trade_history(self, info):
        if not self.trade_history:
            self.trade_history = {key: [] for key in info.keys()}

        for key, value in info.items():
            self.trade_history[key].append(value)


    def render(self, mode='human', close=False):
        """
        Renders a plot with trades made by the agent.
        """
        action_types, action_amounts, prices = zip(*self._action_history)

        def _plot_actions(action, candle, prices):
            color = None
            if action == 'Sell':
                color = 'red'
            elif action == 'Buy':
                color = 'green'
            if color:
                plt.scatter(candle, prices[candle], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            # Plot the prices
            plt.plot(prices)
            
            start_action = action_types[0]

            _plot_actions(start_action, self._start_candle, prices)

        _plot_actions(self._last_action_type, self._current_candle, prices)

        plt.suptitle(
            "Accumulated Reward: %.6f" % self._total_reward_accumulated + ' ~ ' +
            "Net Worth: %.6f" % self._net_worth
        )

        plt.pause(0.01)

    def _process_data(self, df):
        """
        Processes the dataframe into features.
        """
        data_frame = df.iloc[:, 1:] # drop first column which is date

        # Convert df to numpy array
        return data_frame.to_numpy(dtype=np.float32)

    def _generate_action_data_tuple(self, action, price):
        """
        Returns the action type and amount as tuple
        """

        action_type_name = self._get_human_readable_action(action)

        action_type = action_type_name
        amount = action[1]
        return (action_type, amount, price)

    def _get_human_readable_action(self, action):
        action_type_name = 'Buy' if action[0] < 1 else 'Sell' if action[0] < 2 else 'Hold'
        return action_type_name    