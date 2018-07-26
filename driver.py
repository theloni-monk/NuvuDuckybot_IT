#!/usr/bin/python
# Import Adafruit Motor HAT Library
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
# Import additional libraries that support MotorHAT
import time
import atexit
import vector
from evdev import InputDevice, categorize, ecodes, KeyEvent, list_devices

# Get the name of the Logitech Device


def getInputDeviceByName(name):
    devices = [InputDevice(fn) for fn in list_devices()]
    for device in devices:
        if device.name == name:
            return InputDevice(device.fn)
    return None

class Driver:
    def __init__(self, **kwargs):
        # create a default MotorHAT object, no changes to I2C address or frequency
        self.mh = Adafruit_MotorHAT(addr=0x60)
        self.lmotor = mh.getMotor(kwargs.get("motorLeft", 1))
        self.rmotor = mh.getMotor(kwargs.get("motorRight", 4))
        if kwargs.get("enableController", False):
            self.gamepad = getInputDeviceByName(
                kwargs.get("deviceName", "Logitech Gamepad F710"))

    def runMotor(self, motor, speed):
        """ motor - the motor object to control.
            speed - a number from -32768 (reverse) to 32768 (forward) """
        # COMPLETE THIS FUNCTION!
        if 1 <= speed <= 32768:
            motor.run(Adafruit_MotorHAT.FORWARD)
            motor.setSpeed(int(speed*(255/32768)))
        elif speed > 32768:
            motor.run(Adafruit_MotorHAT.FORWARD)
            motor.setSpeed(255)
        elif -1 < speed < 1:
            motor.setSpeed(0)
            mh.BRAKE
        elif -32768 <= speed <= -1:
            motor.run(Adafruit_MotorHAT.BACKWARD)
            motor.setspeed(int(-speed*(255/32768)))
        elif speed < -32768:
            motor.run(Adafruit_MotorHAT.BACKWARD)
            motor.setspeed(255)

    def runMotorNorm(self, motor, speed):
        return self.runMotor(motor, speed/32768)

    def runDebug(self):
        self.runMotor(self.lmotor, 32767)
        self.runMotor(self.rmotor, 32767)

    def runAngle(self, vector, speed=1):
        vector *= speed
        self.runMotorNorm(self.motorl,vector[0])
        self.runMotorNorm(self.motorr,vector[1])

    def controllerOverride(self, **kwargs):
        """ Blocking: use for debug/override only """
        for event in gamepad.read_loop():
            if event.type == ecodes.EV_KEY:
                keyevent = categorize(event)
                if keyevent.keystate == KeyEvent.key_down:
                    print(keyevent.keycode)
                # example key detection code
                if 'BTN_A' in keyevent.keycode:
                    # Do something here when the A button is pressed
                    pass
                elif 'BTN_START' in keyevent.keycode:
                    # Do something here when the START button is pressed
                    pass
            elif event.type == ecodes.EV_ABS:
                if event.code == 0:
                    print('PAD_LR '+str(event.value))
                elif event.code == 1:
                    print('PAD_UD '+str(event.value))
                elif event.code == 2:
                    print('TRIG_L '+str(event.value))
                elif event.code == 3:
                    print('JOY_LR '+str(event.value))
                elif event.code == 4:
                    print('JOY_UD '+str(event.value))
                elif event.code == 5:
                    print('TRIG_R '+str(event.value))
                elif event.code == 16:
                    print('HAT_LR '+str(event.value))
                elif event.code == 17:
                    print('HAT_UD '+str(event.value))
                else:
                    pass

# recommended for auto-disabling motors on shutdown!


def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)


atexit.register(turnOffMotors)

if __name__ == "__main__":
    driver = Driver(enableController=True)
    driver.controllerOverride()
