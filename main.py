import pygame
import config
from environment import Environment
from agent import Agent
from utils import log_result, plot_rewards

pygame.init()
screen = pygame.display.set_mode((500, 500))
clock = pygame.time.Clock()

CELL_SIZE = 5 #a single grid cell.
WINDOW_SIZE = 50 #determines how many recent episodes are considered when evaluating learning progress.
SUCCESS_THRESHOLD = 0.7 #minimum success rate for environment difficulty
OBSTACLE_STEP = 50 #how many obstacles added
INITIAL_OBSTACLES = config.INIT_OBSTACLES #starting number of obstacles

success_history = []
current_obstacle_count = INITIAL_OBSTACLES


def draw_cell(x, y, color):#a colored rectangle at coordinates
    pygame.draw.rect(
        screen,
        color,
        (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )


def draw_grid():#Light gray grid lines
    size = config.GRID_SIZE * CELL_SIZE
    for i in range(config.GRID_SIZE + 1):
        pygame.draw.line(screen, (200, 200, 200),
                         (i * CELL_SIZE, 0),
                         (i * CELL_SIZE, size), 1)
        pygame.draw.line(screen, (200, 200, 200),
                         (0, i * CELL_SIZE),
                         (size, i * CELL_SIZE), 1)


def draw_env():
    screen.fill((255, 255, 255))#Clears the screen

    for ox, oy in env.obstacles:#Draws obstacles
        draw_cell(ox, oy, (33, 66, 30))

    gx, gy = env.goal#Draws the goal position
    draw_cell(gx, gy, (255, 0, 0))

    for agent in agents:#Draws all agents
        draw_cell(agent.pos[0], agent.pos[1], (251, 174, 210))

    draw_grid()



env = Environment()
env.generate_obstacles(current_obstacle_count)#Obstacles are randomly generated.
env.reset_goal()#random goal point

agents = [Agent(config.GRID_SIZE, env.obstacles)#All agents share the same environment and obstacles.
          for _ in range(config.NUM_AGENTS)]

episode = 1
running = True



while running and episode <= config.EPISODES:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

  
    for agent in agents:#Each agent is placed in a random
        agent.reset()

    total_rewards = {agent: 0 for agent in agents}#reward for every agent(step , collısıon , goal)
    agent_done = {agent: False for agent in agents}#agent reaches the target location and completed

   
    for step in range(config.MAX_STEPS):#each episode finish at max_step

        for agent_id, agent in enumerate(agents):

            if agent_done[agent]:
                continue

            old_pos = agent.pos
            action = agent.choose_action()#The agent selects an action using an ε-greedy policy

            other_agents = {a.pos for a in agents if a != agent}
            new_pos, reward, done = env.step(old_pos, action, other_agents)#When an agent reaches the goal, it is stopped.

            agent.pos = new_pos
            agent.update_q(old_pos, action, reward, new_pos)#The Q-table is updated using the standard Q-learning update rule

            
            total_rewards[agent] += reward#This line adds the reward the agent receives at each step to its total reward.

            if done:
                agent_done[agent] = True#This check determines whether the agent reached the target.

        draw_env()
        pygame.display.update()
        clock.tick(20)

   
    episode_success = 0

    for agent_id, agent in enumerate(agents):

        if agent_done[agent]:
            status = "ReachedGoal"#agent reached the goal
            episode_success += 1
        else:
            status = "Collision"#else collision

        log_result( 
            episode,
            agent_id,
            total_rewards[agent], 
            status,
            agent.epsilon
        )

    
    success = 1 if episode_success > 0 else 0 #If the agent is successful, this is retained .
    success_history.append(success)

    if len(success_history) > WINDOW_SIZE:#If the success rate for the specified episode exceeds 70, the number of obstacles is increased.
        success_history.pop(0)#if success_history full delete the first one

    
    for agent in agents:
        agent.decay_epsilon()#This code block is used to reduce the exploration rate (ε) of all agents at the end of each episode.

    
    if len(success_history) == WINDOW_SIZE:
        success_rate = sum(success_history) / WINDOW_SIZE 
        if success_rate >= SUCCESS_THRESHOLD: # if success rate > 70
            current_obstacle_count += OBSTACLE_STEP # add new obstacles
            env.generate_obstacles(current_obstacle_count)#create environment again
            env.reset_goal()#create new goal
            success_history.clear()

    episode += 1


plot_rewards()
pygame.quit()
