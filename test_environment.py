import time
from rlbotenv import *
from qbot_virtual import *
from renderer import Renderer

################################################################################
#                                                                              #
# Sample: static test of virtual robot training environment                    #
#                                                                              #
################################################################################

r = Renderer()

# make a bot and some obstacles (change values to see what happens...)
bot = QvBot(sensor_sectors=5,degrees_per_sensor_sector=45,turn_sectors=8)
obstacles = [Obstacle(200,0,25)]
env = RlBotEnv(bot)
r.render_backdrop(obstacles,bot,True)  # draw a picture of obstacles

print(env.reset(obstacles=obstacles))
print(env.step(0))
print(env.step(1))
print(env.step(2))

while True:
    time.sleep(1)   # don't use pass!... that would tie up the CPU
