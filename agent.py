import numpy as np
import config
import random

class Agent:

    def __init__(self, grid_size, obstacles):
        self.grid_size = grid_size#For the starting position, the Agent knows the boundaries of the environment and the forbidden cells.
        self.obstacles = obstacles
        self.q_table = np.zeros((grid_size, grid_size, 4))#Initially, the agent is considered “unaware” in all situations and for all actions.
        self.epsilon = config.EPSILON_START#Epsilon decay
        self.reset()#The agent is placed in a valid cell.

    def reset(self):
        while True:
            pos = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if pos not in self.obstacles:
                self.pos = pos#If it is on the obstacle, a new position is determined.
                break

    def choose_action(self):
        if random.random() < self.epsilon:#A random action is selected:with a probability of %ε
            return random.randint(0, 3)#A random action is selected
        x, y = self.pos#Exploitation
        return np.argmax(self.q_table[x, y])#The Q values at the current position are obtained. The action with the highest Q is selected.
   
    #Q-Learning Update
    def update_q(self, old_pos, action, reward, new_pos):
        x1, y1 = old_pos
        x2, y2 = new_pos

        old_q = self.q_table[x1, y1, action]#Current Q Value
        max_next = np.max(self.q_table[x2, y2])#Best Future Q Value

        self.q_table[x1, y1, action] = old_q + config.ALPHA * (
            reward + config.GAMMA * max_next - old_q
        )#Q-Learning Update Formula

    def decay_epsilon(self):
         self.epsilon = max(
            config.EPSILON_MIN,
            self.epsilon * config.EPSILON_DECAY
        )

