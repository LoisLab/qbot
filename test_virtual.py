import time
from rlbotenv import *
from qbot_virtual import *
from renderer import Renderer

################################################################################
#                                                                              #
# Sample: static test of virtual robot                                         #
#                                                                              #
################################################################################

r = Renderer()

# make a bot and some obstacles (change values to see what happens...)
bot = QvBot(sensor_sectors=5,degrees_per_sensor_sector=45,turn_sectors=8)
obstacles = [Obstacle(200,0,25),Obstacle(-200,0,25),Obstacle(0,200,25),Obstacle(0,-200,25)]
r.render_backdrop(obstacles,bot,True)  # draw a picture of obstacles

print('-- Initial --------------------------------------------')
print('init  :', bot.get_observation(obstacles), bot)

for action in range(bot.action_space()):
    print('--- Action ', action, '-----------------------------------------')
    print('before:', bot.get_observation(obstacles), bot)
    bot.move(action)
    print('after :', bot.get_observation(obstacles), bot)
    r.render_step(bot)

while True:
    time.sleep(1)   # don't use pass!... that would tie up the CPU
