class Episode:
    def __init__(self, episode, day, state, action, reward):
        self.episode = episode
        self.day = day
        self.state = state
        self.action = action
        self.reward = reward