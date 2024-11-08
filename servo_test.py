#Here we will import all the extra functionality desired
from time import *
from adafruit_servokit import ServoKit

#Below is an initialising statement stating that we will have access to 16 PWM channels of the HAT and to summon them we will use | kit |
kit = ServoKit(channels=16)

kit.servo[0].angle = None

while True:
    sleep(1)
    kit.servo[0].angle = 0
    sleep(1)
    kit.servo[0].angle = 90
    sleep(1)
    kit.servo[0].angle = 180
