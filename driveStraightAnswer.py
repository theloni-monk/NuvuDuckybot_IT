#!/usr/bin/python
# Import Adafruit Motor HAT Library
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
# Import additional libraries that support MotorHAT
import time
import atexit

# create a default MotorHAT object, no changes to I2C address or frequency
mh = Adafruit_MotorHAT(addr=0x60)
lmotor = mh.getMotor(1)
rmotor = mh.getMotor(4)

# recommended for auto-disabling motors on shutdown!
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)

def runMotor(motor, speed):
    """ motor - the motor object to control.
        speed - a number from -32768 (reverse) to 32768 (forward) """
    # COMPLETE THIS FUNCTION!
    if 1 <= speed <= 32768:
        motor.run(Adafruit_MotorHAT.FORWARD)
        motor.setSpeed(int(speed*(255/32768)))
    elif speed>32768:
        motor.run(Adafruit_MotorHAT.FORWARD)
        motor.setSpeed(255)
    elif -1 < speed < 1:
        motor.setSpeed(0)
        mh.BRAKE
    elif -32768 <= speed <= -1:
        motor.run(Adafruit_MotorHAT.BACKWARD)
        motor.setspeed(int(-speed*(255/32768)))
    elif speed<-32768:
        motor.run(Adafruit_MotorHAT.BACKWARD)
        motor.setspeed(255)

#drive both motors straight for debuging
runMotor(lmotor, 32767)
runMotor(rmotor, 32767)
