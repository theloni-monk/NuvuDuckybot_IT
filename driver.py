#!/usr/bin/python
# Import Adafruit Motor HAT Library
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor
# Import additional libraries that support MotorHAT
import time
import atexit
from evdev import InputDevice, categorize, ecodes, KeyEvent, list_devices
import math
# Get the name of the Logitech Device


def getInputDeviceByName(name):
    devices = [InputDevice(fn) for fn in list_devices()]
    for device in devices:
        if device.name == name:
            return InputDevice(device.fn)
    return None

def normVector(vect):
    Ovect=[0,0]
    mag=math.sqrt(vect[0]**2+vect[1]**2)
    Ovect[0],Ovect[1]=vect[0]/mag,vect[1]/mag
    return Ovect

class Driver:
    def __init__(self, **kwargs):
        # create a default MotorHAT object, no changes to I2C address or frequency
        self.mh = Adafruit_MotorHAT(addr=0x60)
        self.lmotor = self.mh.getMotor(kwargs.get("motorLeft", 1))
        self.rmotor = self.mh.getMotor(kwargs.get("motorRight", 4))
        if kwargs.get("enableController", False):
            self.gamepad = getInputDeviceByName(
                kwargs.get("deviceName", "Logitech Gamepad F710"))
        atexit.register(self.turnOffMotors)

    def runMotor(self, motor, speed):
        """ motor - the motor object to control.
            speed - a number from -32768 (reverse) to 32768 (forward) """
        #print(speed)
        if speed>0:
            motor.run(Adafruit_MotorHAT.FORWARD)
        else:
            motor.run(Adafruit_MotorHAT.BACKWARD)
        
        if 1 <= speed <= 32767:       
            motor.setSpeed(int(speed*(255.0/32768.0)))
        elif speed > 32768:
            motor.setSpeed(255)
        elif -1 < speed < 1:
            motor.setSpeed(0)
            self.mh.BRAKE
        elif -32768 <= speed <= -1:
            motor.setSpeed(int(-speed*(255.0/32768.0)))
        elif speed < -32768:
            motor.setSpeed(255)

    def runMotorNorm(self, motor, speed):
        return self.runMotor(motor, speed/32768)

    def runDebug(self):
        self.runMotor(self.lmotor, 32767)
        self.runMotor(self.rmotor, 32767)

    def runDiff(self, diff, speed=32767, Snormed=False):
        if Snormed:
            speed*=32767
        vectO=normVector(diff)
        vectO[0],vectO[1]=vectO[0] * speed, vectO[1]*speed
        print(vectO)
        self.runMotor(self.lmotor, vectO[0])
        self.runMotor(self.rmotor, vectO[1])

    def controllerOverride(self, **kwargs):
        """ Blocking: use for debug/override only """
        # Get the name of the Logitech Device
        def getInputDeviceByName(name):
            devices = [InputDevice(fn) for fn in list_devices()]
            for device in devices:
                if device.name == name:
                    return InputDevice(device.fn)
            return None

        # Import our gamepad.
        gamepad = getInputDeviceByName('Logitech Gamepad F710')        
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
                    self.runDiff
                elif event.code == 3:
                    print('JOY_LR '+str(event.value))
                    self.runDiff([1,-1],event.value)
                
                elif event.code == 4:
                    print('JOY_UD '+str(event.value))
                    #self.runAngle([1,1], event.value)
                elif event.code == 5:
                    print('TRIG_R '+str(event.value))
                    self.runDiff([1,1])

                elif event.code == 16:
                    print('HAT_LR '+str(event.value))
                elif event.code == 17:
                    print('HAT_UD '+str(event.value))
                else:
                    pass

# recommended for auto-disabling motors on shutdown!
    def turnOffMotors(self):
        self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)



if __name__ == "__main__":
    driver = Driver(enableController=True)
    driver.controllerOverride()
