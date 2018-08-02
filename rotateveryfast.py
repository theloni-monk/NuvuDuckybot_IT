from driver import Driver # use the legacy module it's pretty great
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
if __name__ == "__main__":
    driver = Driver()
    motor = driver.lmotor
    motor.run(Adafruit_MotorHAT.FORWARD)
    motor.setSpeed(255)