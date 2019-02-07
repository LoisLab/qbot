import math
import numpy as np
from obstacle import Obstacle

################################################################################
#                                                                              #
# Learning environment for Q-Bot:                                              #
#    - creates discrete observations from continous sensor readings:           #
#         - sensor = [1000,1200,1110,90,1150,880]                              #
#         - observation = argmin(senor) = 3 (the index of the lowest reading)  #
#    - calculates reward based on change in distance to nearest object         #
#    - relies on robot to provide sensor readings, action space, actions       #
#                                                                              #
################################################################################

class RlBotEnv:

    MAX_ITERATIONS = 500
    MAX_REWARD = 100

    def __init__(self, bot):
        self.bot = bot        # could be a physical or virtual robot

    def get_discrete_observation(self, obs):
        if min(obs) >= self.bot.max_sensor_range(): # anything in range?
            return len(obs)                         # nothing detected
        else:
            return np.argmin(obs)                   # closest object

    def reset(self, obstacle_count=0, obstacles=None):
        self.iterations = 0
        self.bot.reset()
        if obstacles is None:
            self.obstacles = Obstacle.make_obstacles(obstacle_count)
        else:
            self.obstacles = obstacles
        obs = self.bot.get_observation(self.obstacles)
        self.min_distance = min(obs)              # for use in first call to step()
        return self.get_discrete_observation(obs) # return discrete observation

    def step(self, action):
        self.iterations += 1
        self.bot.move(action)
        obs = self.bot.get_observation(self.obstacles)
        reward = self.min_distance-min(obs) # reward = reduction in distance
        reward = max(min(reward, RlBotEnv.MAX_REWARD), -RlBotEnv.MAX_REWARD)
        self.min_distance = min(obs)        # for use in next call to step()
        state = self.get_discrete_observation(obs)   # convert to discrete state
        done = min(obs) < self.bot.goal() or self.iterations == RlBotEnv.MAX_ITERATIONS
        return state, reward, done
