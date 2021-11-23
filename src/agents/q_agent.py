import numpy as np

class QAgent:
    def __init__(self, states, actions, qtable=None, exploration_ratio=0.1, learning_rate=0.2, discount_factor=0.9):
        # Set parameters of the agent
        self.exploration_ratio = exploration_ratio
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.actions = actions

        # Initialize Q-table
        if qtable is not None:
            self.qtable = qtable
        else:
            self.qtable = np.zeros((states.n, actions.n))

    def get_next_step(self, state):
        # Exploration
        if np.random.rand() < self.exploration_ratio:
            action = self.actions.sample()
        # Exploitation
        else:
            random_values = np.random.uniform(low=0, high=1, size=(1, self.actions.n))/1000
            action = np.argmax(self.qtable[state]+random_values)
        # e-greedy decay
        if self.exploration_ratio > 0.05:
            self.exploration_ratio -= 0.01
        return action

    def update_qtable(self, state, action, reward, next_state):
        # Update Q-table if not final state
        if reward == 0:
            self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (reward + self.discount_factor * np.max(self.qtable[next_state]) - self.qtable[state, action])
        # Update Q-table if final state
        else:
            self.qtable[state, action] = self.qtable[state, action] + self.learning_rate * (reward - self.qtable[state, action])
    
    def get_qtable(self):
        return self.qtable

    def print_qtable(self):
        print(self.qtable)