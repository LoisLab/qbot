from stepper import Stepper
import numpy as np
import VL53L1X
import math
import time

################################################################################
#                                                                              #
# Distance Sensor using SparkFun                                               #
#   see: https://www.sparkfun.com/products/14722                               #
#                                                                              #
################################################################################

class DistanceSensor:

    pins = (19,21,23,24)

    def __init__(self, motor):
        self.position = 0.0  # 0 is forward
        self.lidar = VL53L1X.VL53L1X(i2c_bus=1, i2c_address=0x29)
        self.lidar.open() # Initialise the i2c bus and configure the sensor
        self.motor = motor

    def get_reading(self, position):
        self.seek_position(position)
        return self.motor.get_position(), self.read_distance()

    def seek_position(self, target):
        self.motor.set_target(target)
        while self.motor.has_steps():
            self.motor.do_step()

    def read_distance(self):
        distance_in_mm = 0
        self.lidar.start_ranging(3) # Start ranging, 1 = Short Range, 2 = Medium Range, 3 = Long Range
        for n in range(5):
            distance_in_mm += self.lidar.get_distance()
        distance_in_mm /= 5.0
        self.lidar.stop_ranging() # Stop ranging
        return distance_in_mm

    def max_sensor_range(self):
        return 4000 # 4 meters
