import numpy as np
from rlbotenv import *
from qbot_virtual import *
from renderer import Renderer

def one_hot(n,m):
    return [1 if x==n else 0 for x in range(m)]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def policy(obs,weights,explore=0.1):
    if np.random.random()<explore:
        return np.random.randint(np.shape(weights)[1])
    else:
        input_vector = np.dot(obs,weights)
        output_vector = sigmoid(input_vector)
        return np.argmax(output_vector)

env = RlBotEnv(QvBot(sensor_sectors=5,degrees_per_sensor_sector=22.5,turn_sectors=8))
r = Renderer(100)

weights = np.random.random((env.bot.observation_space(),env.bot.action_space()))
print(np.shape(weights))
for m in range(1):
    memory = []
    for n in range(1):
        memory.append([])
        obs = env.reset(obstacle_count=1)
        done = False
        while not done:
            action = policy(obs,weights)
            obs,reward,done = env.step(action)
            memory[n].append([obs,action,reward])
    for episode in memory:
        reward = 0
        for step in reversed(episode):
            reward += step[2]
            step.append(reward)
            reward *= 0.9
    print(memory)
