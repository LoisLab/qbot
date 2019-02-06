from rlbotenv import *
from qbot_virtual import *
from renderer import Renderer

################################################################################
#                                                                              #
# Basic Q-Learning: Virtual Robot                                              #
#                                                                              #
################################################################################

env = RlBotEnv(QvBot(sensor_sectors=5,degrees_per_sensor_sector=22.5,turn_sectors=8))
r = Renderer(100)

# create the q-table
q = np.zeros((env.bot.observation_space(), env.bot.action_space()))

# try changing these hyper-parameters...
explore = 0.1   # exploration rate (odds of taking a random action)
alpha = 0.01    # learning rate (proportional weight of new v. old information)
gamma = 0.6     # discount rate (relative value of future v. current reward)

try:
    for n in range(1000000):
        state = env.reset(obstacle_count=1)
        r.render_reset(env, raytrace=True)
        done = False
        steps = 0
        while not done:
            steps += 1
            if np.random.random() < explore:   # explore the state-action space
                action = env.bot.sample()        # ...random action
            else:                              # exploit the info in the q-table
                action = np.argmax(q[state])   # ...best known action
            next_state, reward, done = env.step(action)
            # update the q-table (see https://en.wikipedia.org/wiki/Q-learning)
            q[state][action] = (1-alpha)*q[state][action] + alpha*(reward+gamma*np.max(q[next_state]))
            state = next_state
            r.render_step(env.bot)
        print(n)

except KeyboardInterrupt:
    print(q)
