import numpy as np
import random
import itertools
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

from episode import Episode

# Definicion de ambiente
class InventoryEnvironment:

    def __init__(self):
        self.products = ['product_A', 'product_B']
        self.max_stock = 10
        self.demand = {'product_A': [0,1,2], 'product_B': [0,1,2]}
        self.restock_cost = {'product_A': 5, 'product_B': 7}
        self.sell_price = {'product_A': 10, 'product_B': 15}
        self.state = None

    def reset(self):
        self.state = {product: random.randint(0, self.max_stock) for product in self.products}
        return self.state

    def step(self, action):
        reward = 0
        for product in self.products:
            stock = self.state[product]
            restock = action[product]
            self.state[product] = min(self.max_stock, stock + restock)
            demand = random.choice(self.demand[product])
            sales = min(demand, self.state[product])
            reward += sales * self.sell_price[product] - restock * self.restock_cost[product]

        return self.state, reward

    def generate_episode(self, env, policy, episode, total_days, action=None):
        episode_info = []
        state = tuple(env.reset().values())

        for day in range(total_days):
            if action is None: action = policy[state]
            new_state, reward = env.step(action)

            episode_obj = Episode(episode, day, state, action.copy(), reward)
            episode_info.append(episode_obj)
            
            state = tuple(new_state.values())

        return episode_info

    def generate_arbitrary_policies(self):
        policy = {}
        stock_levels = range(self.max_stock + 1)
        all_states = list(itertools.product(stock_levels, repeat=len(self.products)))

        restock_amount = 5
        for state_tuple in all_states:
            policy[state_tuple] = {product: restock_amount for product in self.products}

        return policy

    def mc_exploring_starts(self, env, total_episodes, total_days, gamma=0.9):
        # Initialize:
        pi = self.generate_arbitrary_policies()
        Q = defaultdict(lambda: defaultdict(float))
        returns = defaultdict(list)

        for episode in range(total_episodes):
            # choosing random initial state and action
            init_state = random.choice(list(pi.keys()))
            init_action = pi[init_state]

            episode_data = env.generate_episode(env, pi, episode, total_days, action=init_action)
            
            G = 0
            visited_state_action = set()
            for current_step in reversed(range(len(episode_data))):
                state = episode_data[current_step].state
                action = episode_data[current_step].action
                reward = episode_data[current_step].reward
                G = gamma * G + reward
                action_tuple = tuple(action.items())

                if (state, action_tuple) not in visited_state_action:
                    visited_state_action.add((state, action_tuple))
                    
                    # Storing the return
                    returns[(state, action_tuple)].append(G)
                    
                    # Updating Q
                    Q[state][action_tuple] = np.mean(returns[(state, action_tuple)])
                    
                    # Updating policy
                    best_action = max(Q[state], key=Q[state].get)
                    pi[state] = dict(best_action)
        
        return pi, Q

env = InventoryEnvironment()
policy, Q = env.mc_exploring_starts(env, total_episodes=2000, total_days=30)