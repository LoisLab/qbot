import numpy as np
from rlbotenv import *
from qbot_virtual import *
from renderer import Renderer
import matplotlib.pyplot as plt

################################################################################
#                                                                              #
# Rudimentary policy gradient algorithm tuned for amusing imperfection         #
#                                                                              #
################################################################################

def one_hot(m,n):
    return [1 if x==m else 0 for x in range(n)]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sig_prime(sig):
    return 1*(1-sig)

def policy(obs,input_size,weights):
    one_hot_obs = one_hot(obs, input_size)
    input_vector = np.dot(one_hot_obs,weights)
    output_vector = sigmoid(input_vector)
    return input_vector, output_vector, np.argmax(output_vector)

def sample(memory):
    episode = memory[np.random.randint(n+1)]
    return episode[np.random.randint(len(episode))]

env = RlBotEnv(QvBot(sensor_sectors=1,degrees_per_sensor_sector=12,turn_sectors=180))

alpha = 0.01
gamma = 0.99

input_size = env.bot.observation_space()
output_size = env.bot.action_space()
weights = np.random.random((input_size,output_size))*2-1
print('dimensions of net: ', np.shape(weights))

for m in range(10):
    print(m)
    memory = []
    for n in range(10):
        memory.append([])
        state = env.reset(obstacle_count=1)
        done = False
        while not done:
            action = policy(state,input_size,weights)[2]
            obs,reward,done = env.step(action)
            memory[n].append([state,action,reward])
            state = obs

    for episode in memory:
        reward = 0
        for step in reversed(episode):
            reward += step[2]
            step.append(reward)
            reward *= gamma

    for i in range(10):
        s = sample(memory)
        state = s[0]       # integer state
        action = s[1]      # integer action
        fwd_reward = s[3]  # cumulative (discounted) reward starting from (s,a)
        policy_output = policy(state,input_size,weights)[1]
        policy_prime = sig_prime(policy_output)
        weights[state][action] = (1-alpha) * weights[state][action] + alpha*policy_prime[action]*fwd_reward

try:
    r = Renderer()
    for trial in range(10):
        state = env.reset(obstacle_count=1)
        r.render_reset(env, raytrace=True)
        done = False
        while not done:
            action = policy(state,input_size,weights)[2]
            state, reward, done = env.step(action)
            r.render_step(env.bot)

except KeyboardInterrupt:
    print(weights)
