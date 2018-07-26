#!/usr/bin/python
# Import Adafruit Motor HAT Library
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
# Import additional libraries that support MotorHAT
import time
import atexit

# create a default MotorHAT object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT(addr=0x60)
lmotor = mh.getMotor(1)
rmotor = mh.getMotor(2)

# recommended for auto-disabling motors on shutdown!


def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)


# Complete this function so:
# 1. values in the range 1 to 32768 make the motor spin forward faster and faster.
# 2. values in the range -1 to -32768 make the motor spin backward faster and faster.
# 3. any value equal to 0 makes the motor BRAKE.
# 4. any values less than -32768 and greater than 32768 use the max speed in the right direction.
def runMotor(motor, speed):
    """ motor - the motor object to control.
        speed - a number from -32768 (reverse) to 32768 (forward) """
    # COMPLETE THIS FUNCTION!
    if 1 <= speed <= 32768:
        mh.forward
        motor.setSpeed(int(speed*(255/32768)))
    elif -1 < speed < 1:
        motor.setSpeed(0)
        mh.BRAKE
    elif -32768 <= speed <= -1
        mh.backward
        motor.setspeed(int(-speed*(255/32768)))


runMotor(lmotor, 32767)
runMotor(rmotor, 32767)
