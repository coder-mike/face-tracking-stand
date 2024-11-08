#Here we will import all the extra functionality desired
from time import *
from adafruit_servokit import ServoKit

#Below is an initialising statement stating that we will have access to 16 PWM channels of the HAT and to summon them we will use | kit |
kit = ServoKit(channels=16)

#Below desides the initial angle that the servo which is attatched to Port 0 will be. In this case we will make it zero degrees.
kit.servo[0].angle = None