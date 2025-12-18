import random
import config


ACTIONS = {
    0: (0, -1),#Up
    1: (0, 1),#Down
    2: (-1, 0),#Left
    3: (1, 0)#Right
}

class Environment:

    def __init__(self):
        self.grid_size = config.GRID_SIZE
        self.generate_obstacles()
        self.reset_goal()

    def reset_goal(self):#Produces a single target at the beginning of the episode
        
        while True:
            g = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if g not in self.obstacles: #Prevent the goal from coming onto an obstacle
                self.goal = g
                break

    def generate_obstacles(self, obstacle_count=300):
      self.obstacles = set()#Prevents adding the same coordinates again
      while len(self.obstacles) < obstacle_count:
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        self.obstacles.add((x, y))#A random cell block is created


    def step(self, pos, action, other_agents):#pos: The agent's current position (x, y)
                                              #action: Selected action (0–3)
        dx, dy = ACTIONS[action]    #Action is converted into a vector          
        new_pos = (pos[0] + dx, pos[1] + dy)#Calculate new position

        #Cannot exit the grid. You will be penalized.
        if not (0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size):
            return pos, config.REWARD_COLLISION, False #The agent stays in the old position

        #If an obstacle cell is entered
        if new_pos in self.obstacles:
            return pos, config.REWARD_COLLISION, False#Action canceled. You will be penalized.

        
        # Collision with other agents (except goal cell)
        if new_pos in other_agents and new_pos != self.goal:
           return pos, config.REWARD_COLLISION, False

        # Goal
        if new_pos == self.goal:
            return new_pos, config.REWARD_GOAL, True #Reward goal.The episode ends.

        return new_pos, config.REWARD_STEP, False
