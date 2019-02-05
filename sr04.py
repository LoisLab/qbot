import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)

class Sr04():
    def __init__(self, trigger_pin, echo_pin):
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.distance = 0
        self.starttime = 0.0
        self.stoptime = 0.0
        GPIO.setup(self.trigger_pin, GPIO.OUT)
        GPIO.setup(self.echo_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

    
    def read_distance(self):
        GPIO.output(self.trigger_pin, True)           # pulse to start a reading 
        time.sleep(0.00001)
        GPIO.output(self.trigger_pin, False)
        while GPIO.input(self.echo_pin) == 0:       # catch time until state changes
            StartTime = time.time()
        while GPIO.input(self.echo_pin) == 1:
            StopTime = time.time()
        TimeElapsed = StopTime - StartTime      # calculate time
        self.distance = (TimeElapsed * 34300) / 2    # 34300 cm/s round trip - halve to get distance to target
        return self.distance

    def __str__(self):
        return str(self.distance)
        
