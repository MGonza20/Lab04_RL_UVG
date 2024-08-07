import numpy as np
import random
import itertools
from collections import defaultdict

from episode import Episode
import matplotlib.pyplot as plt


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

		def generate_episode(self, policy, total_days, epsilon=0.1):
				episode_info = []
				state = tuple(self.reset().values())

				for day in range(total_days):
						selected_action = policy(state, epsilon)

						new_state, reward = self.step(selected_action)

						episode_obj = Episode(day, day, state, selected_action, reward)
						episode_info.append(episode_obj)
						
						state = tuple(new_state.values())

				return episode_info

		def generate_policies(self):
				policy = {}
				stock_levels = range(self.max_stock + 1)
				all_states = list(itertools.product(stock_levels, repeat=len(self.products)))

				for state_tuple in all_states:
						policy[state_tuple] = {product: random.randint(0, env.max_stock) for product in env.products}

				return policy
		
		def epsilon_greedy_policy(self, Q, state, epsilon):
						if random.random() < epsilon:
								return {product: random.randint(0, self.max_stock) for product in self.products}
						else:
								best_action = max(Q[state], key=Q[state].get, default={product: 0 for product in self.products})
								return dict(best_action)
		
		def random_policy(self, state, epsilon=None):
				return {product: random.randint(0, self.max_stock) for product in self.products}

		def mc_exploring_starts(self, total_episodes, total_days, gamma=0.9, epsilon=0.1):
				Q = defaultdict(lambda: defaultdict(float))
				returns = defaultdict(list)
				for episode in range(total_episodes):
						episode_data = self.generate_episode(lambda state, epsilon: self.epsilon_greedy_policy(Q, state, epsilon), total_days, epsilon)

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
										
										returns[(state, action_tuple)].append(G)
										
										Q[state][action_tuple] = np.mean(returns[(state, action_tuple)])
										
				return Q
		
		def mc_off_policy(self, total_episodes, total_days, gamma=0.9, epsilon=0.1):
				Q = defaultdict(lambda: defaultdict(float))
				returns = defaultdict(list)
				C = defaultdict(lambda: defaultdict(float)) 
				for episode in range(total_episodes):
						# Generar un episodio usando la política de comportamiento (random)
						episode_data = self.generate_episode(self.random_policy, total_days)

						G = 0
						W = 1
						for current_step in reversed(range(len(episode_data))):
								state = episode_data[current_step].state
								action = episode_data[current_step].action
								reward = episode_data[current_step].reward
								G = gamma * G + reward
								action_tuple = tuple(action.items())

								# Actualizar el acumulador de importancia ponderada
								C[state][action_tuple] += W

								# Actualizar Q usando la importancia ponderada
								Q[state][action_tuple] += (W / C[state][action_tuple]) * (G - Q[state][action_tuple])

								# Actualizar W
								pi_action_prob = 1 if self.epsilon_greedy_policy(Q, state, epsilon) == action else 0
								mu_action_prob = 1 / (self.max_stock + 1)  # Probabilidad uniforme bajo política aleatoria

								W *= pi_action_prob / mu_action_prob

								# Si W es 0, salir del bucle
								if W == 0:
										break

				return Q

env = InventoryEnvironment()
print("Usando Off Policy: ")
Q = env.mc_off_policy(total_episodes=2000, total_days=30, epsilon=0.1)
# Generar un episodio usando la política epsilon-greedy
total_days = 30
epsilon = 0.1
episode_data = env.generate_episode(lambda state, epsilon: env.epsilon_greedy_policy(Q, state, epsilon), total_days, epsilon)

# Extraer los niveles de stock de cada producto a lo largo del episodio
days = list(range(total_days))
stock_product_A = [step.state[0] for step in episode_data]
stock_product_B = [step.state[1] for step in episode_data]

# Graficar los niveles de stock
plt.figure(figsize=(12, 6))
plt.plot(days, stock_product_A, label='Product A', marker='o')
plt.plot(days, stock_product_B, label='Product B', marker='o')
plt.title('Cambio de Stock durante el Episodio')
plt.xlabel('Días')
plt.ylabel('Niveles de Stock')
plt.legend()
plt.grid(True)
plt.show()


# Graficar el impacto de epsilon en la recompensa acumulada
epsilon_values = [0.9, 0.5, 0.1]
total_episodes = 2000
total_days = 30

rewards_by_epsilon = {}

for epsilon in epsilon_values:
    cumulative_rewards = []
    for episode in range(total_episodes):
        episode_data = env.generate_episode(lambda state, epsilon=epsilon: env.epsilon_greedy_policy(Q, state, epsilon), total_days, epsilon)
        
        total_reward = sum(step.reward for step in episode_data)
        cumulative_rewards.append(total_reward)
    
    rewards_by_epsilon[epsilon] = cumulative_rewards
plt.figure(figsize=(14, 7))
for epsilon, rewards in rewards_by_epsilon.items():
    plt.plot(range(total_episodes), rewards, label=f'Epsilon = {epsilon}')
    
plt.title('Impact of Epsilon on Cumulative Reward')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.legend()
plt.show()


on_policy_rewards = []
Q_on_policy = env.mc_exploring_starts(total_episodes=total_episodes, total_days=total_days, epsilon=0.1)


for episode in range(total_episodes):
    episode_data = env.generate_episode(lambda state, epsilon=0.1: env.epsilon_greedy_policy(Q_on_policy, state, epsilon), total_days, epsilon=0.1)
    total_reward = sum(step.reward for step in episode_data)
    on_policy_rewards.append(total_reward)

off_policy_rewards = []
Q_off_policy = env.mc_off_policy(total_episodes=total_episodes, total_days=total_days, epsilon=0.1)

for episode in range(total_episodes):
    episode_data = env.generate_episode(env.random_policy, total_days)
    total_reward = sum(step.reward for step in episode_data)
    off_policy_rewards.append(total_reward)
plt.figure(figsize=(14, 7))
plt.plot(range(total_episodes), on_policy_rewards, label='On-Policy (Exploring Starts)', color='blue')
plt.plot(range(total_episodes), off_policy_rewards, label='Off-Policy', color='red')
plt.title('Impact of On-Policy vs Off-Policy Learning')
plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.grid(True)
plt.legend()
plt.show()