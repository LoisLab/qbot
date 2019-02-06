import numpy as np

################################################################################
#                                                                              #
# Base class for robots                                                        #
#    - robot has only 3 actions                                                #
#         - forward, turn left, turn right                                     #
#    - robot must provide an observation of continous distances to objects     #
#         - if readings are 90 degrees apart, observation space = 360/90 = 4   #
#         - if readings are 15 degrees apart, observation space = 360/15 = 24  #
#                                                                              #
################################################################################

class QBot:

    def __init__(self, sensor_sectors, degrees_per_sensor_sector, turn_sectors):
        self.sensor_sectors = sensor_sectors
        self.degrees_per_sensor_sector = degrees_per_sensor_sector
        self.turn_sectors = turn_sectors

    def action_space(self):         # forward, turn left, turn right
        return 3                    # turn left, turn right, move forward

    def observation_space(self):
        return self.sensor_sectors + 1 # the number of sensor readings per sweep
                                       # plus 1 for 'no object detected'

    def sample(self):
        return np.random.randint(self.action_space())
