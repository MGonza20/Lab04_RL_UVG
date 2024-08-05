import numpy as np
import random

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
    
# Init el ambiente
env = InventoryEnvironment()