from qbot import QBot
import time
import math
import numpy as np
import RPi.GPIO as GPIO
from stepper import Stepper, SeekerStepper, RotateStepper
from steppers import Steppers
from distancesensor_28byj import *

################################################################################
#                                                                              #
# Hardware Robot based on Raspberry Pi and 28byj steppers                      #
#   see: https://www.raspberrypi.org/                                          #
#                                                                              #
################################################################################

class QHwBot_28byj(QBot):
    def __init__(self, sensor_sectors=12, degrees_per_sensor_sector=30, turn_sectors=4):
        super().__init__(sensor_sectors, degrees_per_sensor_sector, turn_sectors)
        GPIO.setmode(GPIO.BOARD)
        self.tuning_coeff = 1.555
        self.stepper1 = RotateStepper((7,11,13,15), orientation=Stepper.CCW)
        self.stepper2 = RotateStepper((31,33,35,37),orientation=Stepper.CW)
        self.stepper3 = SeekerStepper((19,21,23,24), orientation=Stepper.CCW)
        self.motors = Steppers((self.stepper1, self.stepper2, self.stepper3))
        self.lidar = DistanceSensor(self.stepper3)

    def move(self, action):
        rotation_degrees = 360/self.turn_sectors * self.tuning_coeff  # turns are proportional to turn sectors defined above
        travel_duration = 90                                          # forward moves are a 90 degree rotation of both wheels
        if action == 0:
            #go forward
            self.stepper1.set_target(travel_duration)
            self.stepper2.set_target(travel_duration)
        elif action == 1:
            #turn left
            self.stepper1.set_target(rotation_degrees)
            self.stepper2.set_target(-rotation_degrees)
        elif action == 2:
            #turn right
            self.stepper1.set_target(-rotation_degrees)
            self.stepper2.set_target(rotation_degrees)
        self.motors.move()

    def get_observation(self,_=None):   # second parameter applies only to virtual robots
        observation = []
        target = self.sensor_sectors//2 * self.degrees_per_sensor_sector
        for i in range(self.sensor_sectors):
            self.stepper3.set_target(target)
            self.motors.move()
            observation.append(self.lidar.get_reading(target)[1])  # 1-th element is distance
            target -= self.degrees_per_sensor_sector
        self.stepper3.set_target(0)
        self.motors.move()
        return np.array(observation)

    def reset(self):
        pass

    def goal(self):
        return 100.0            # millimeters

    def max_sensor_range(self):
        return self.lidar.max_sensor_range(self):
