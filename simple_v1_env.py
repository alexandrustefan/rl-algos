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
INITIAL_BASE_ASSET = 0
OBSERVATION_WINDOW_SIZE = 24 # Probably we should put it as param ?

class SimpleTradingEnv(gym.Env):
    
    metadata = {'render.modes': ['live', 'human', 'none']}
    visualization = None

    def __init__(self, df_scaled, df_normal, trading_fee):
        
        self.df_scaled = df_scaled.reset_index(drop=True)
        self.df_normal = df_normal.reset_index(drop=True)
        self.window_size = OBSERVATION_WINDOW_SIZE
        self.prices, self.features = self._process_data(df_scaled)
        # The shape of the observation is (window_size * features + environment_features) the environment_features are: quote_asset, base_asset, net_worth. The entire observation is flattened in a 1D np array. 
        self.obs_shape = ((OBSERVATION_WINDOW_SIZE * self.features.shape[1] + 3),)


        # Action space
        #self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3.0, 1.0]), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 100])
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
        self._total_reward_accumulated = None
        self.trade_history = None
        self._first_rendering = None
        

    def reset(self):
        self._done = False
        self._current_candle = self._start_candle
        self._quote_asset = INITIAL_QUOTE_ASSET
        self._base_asset = INITIAL_BASE_ASSET 
        self._net_worth = INITIAL_QUOTE_ASSET # at the begining our net worth is the initial quote asset
        self._previous_net_worth = INITIAL_QUOTE_ASSET # at the begining our previous net worth is the initial quote asset
        self._total_reward_accumulated = 0.
        self._first_rendering = True
        self.trade_history = []
        return self._get_observation()

    def _take_action(self, action):
        self._done = False
        current_price = random.uniform(
            self.df_normal.loc[self._current_candle, "low"], self.df_normal.loc[self._current_candle, "high"])

        print('action: ', action)
        print('action type: ', action[0])
        print('action amount: ', action[1])
        action_type = action[0]
        amount = action[1] / 100
        
        if action_type < 1:
            # Buy % assets
            # Determine the maximum amount of quote asset that can be bought
            available_amount_to_buy_with = self._quote_asset / current_price
            # Buy only the amount that agent chose
            assets_bought = available_amount_to_buy_with * amount
            # Update the quote asset balance
            self._quote_asset -= assets_bought * current_price
            # Update the base asset
            self._base_asset += assets_bought
            # substract trading fee from base asset based on the amount bought
            self._base_asset -= self._trading_fee * assets_bought

            # Add to trade history the amount bought if greater than 0
            if assets_bought > 0:
                self.trade_history.append({'step': self._current_candle, 'type': 'Buy', 'amount': assets_bought, 'price': current_price, 'total' : assets_bought * current_price, 'percent_amount': action[1]})
        

        elif action_type < 2:
            # Sell % assets
            # Determine the amount of base asset that can be sold
            amount_to_sell = self._base_asset * amount
            received_quote_asset = amount_to_sell * current_price
            # Update the quote asset
            self._quote_asset += received_quote_asset
            # Update the base asset
            self._base_asset -= amount_to_sell
            
            # substract trading fee from quote asset based on the amount sold
            self._quote_asset -= self._trading_fee * received_quote_asset

            # Add to trade history the amount sold if greater than 0
            if amount_to_sell > 0:
                self.trade_history.append({'step': self._current_candle, 'type': 'Sell', 'amount': amount_to_sell, 'price': current_price, 'total' : received_quote_asset, 'percent_amount': action[1]})

        else:
            # Hold
            self.trade_history.append({'step': self._current_candle, 'type': 'Hold', 'amount': '0', 'price': current_price, 'total' : 0, 'percent_amount': action[1]})


        # Update the current net worth
        self._net_worth = self._base_asset * current_price + self._quote_asset



    def step(self, action):
        """
        Returns the next observation, reward, done and info.
        """
        
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
            current_step = self._current_candle
        )

        self._current_candle += 1

        self._done = self._net_worth <= 0 or self._current_candle >= len(
            self.df_scaled.loc[:, 'open'].values)
        
        return obs, reward, self._done, info


    def _get_observation(self):
        """
        Returns the current observation.
        """
        data_frame = self.features[(self._current_candle - self.window_size):self._current_candle]
        data_frame = data_frame.flatten()

        # Scale all the values to be between -1 and 1
        obs = np.append(data_frame, np.array([self._net_worth / MAX_NET_WORTH , self._quote_asset / MAX_NUM_QUOTE_OR_BASE_ASSET, self._base_asset / MAX_NUM_QUOTE_OR_BASE_ASSET], dtype=np.float32))

        return obs


    def render(self, mode='human', **kwargs):
        """
        Renders a plot with trades made by the agent.
        """
        
        if mode == 'human':
            print(f'Accumulated Reward: {self._total_reward_accumulated} ---- Current Net Worth: {self._net_worth}')
            print(f'Current Quote asset: {self._quote_asset} ---- Current Base asset: {self._base_asset}')
            print(f'Number of trades: {len(self.trade_history)}')
        
            if(len(self.trade_history) > 0):
                print(f'Last Action: {self.trade_history[-1]["type"]} {self.trade_history[-1]["amount"]} assets ({self.trade_history[-1]["percent_amount"]} %) at price {self.trade_history[-1]["price"]}, total: {self.trade_history[-1]["total"]}')
            print(f'--------------------------------------------------------------------------------------')
        elif mode == 'live':
            if self.visualization == None:
                self.visualization = LiveTradingGraph(self.df_normal, kwargs.get('title', None))

            if self._current_candle > OBSERVATION_WINDOW_SIZE:
                self.visualization.render(self._current_candle, self._net_worth, self.trade_history, window_size=OBSERVATION_WINDOW_SIZE)

    def close(self):
        if self.visualization != None:
            self.visualization.close()
            self.visualization = None
         

    def _process_data(self, df_scaled):
        """
        Processes the dataframe into features.
        """

        prices = self.df_scaled.loc[:, 'close'].to_numpy(dtype=np.float32)

        data_frame = df_scaled.iloc[:, 1:] # drop first column which is date TODO: Should be fixed outside of this class

        # Convert df to numpy array
        return prices, data_frame.to_numpy(dtype=np.float32)

    def _generate_action_data_tuple(self, action, price):
        """
        Returns the action type and amount as tuple
        """

        action_type_name = self._get_human_readable_action(action)

        amount = action[1]
        return (action_type_name, amount, price)

    def _get_human_readable_action(self, action):
        # if action is lower than 1 than it's buy action, if action is lower than 2 than it's sell action otherwise it's hold action
        action_type_name = None
        if action[0] < 1:
            action_type_name = 'Buy'
        elif action[0] < 2:
            action_type_name = 'Sell'
        else:
            action_type_name = 'Hold'
        
        return action_type_name    