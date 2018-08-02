#!/usr/bin/python
import motor
import gamepad
# Import library that allows parallel processing
from multiprocessing import Process, Queue
# Import library that ensures an exist function is called when Python stops.
import atexit
# Import OpenCV
import cv2
import time

# Ensure the motorProcess joins and the motors turn off.


def exitFunction():
    # Send a sign to the gamepadProcess to end
    gamepadq.put("exit")
    # Wait for the gamepadProcess to end
    gamepadp.join()
    # Send a sign to the motorProcess to end
    motorq.put("exit")
    # Wait for the motor process to end
    motorp.join()
    # Kill all the motors
    motor.turnOffMotors()
    # Close any openCV windows
    cv2.destroyAllWindows()

##########################################
# MAIN CODE
##########################################


if __name__ == '__main__':
    # Create a queue to communicate to the motor process.
    motorq = Queue()
    # Create a Process that runs the motors, give it the queue.
    motorp = Process(target=motor.motorProcess, args=(motorq,))
    # Start the motorPorcess
    motorp.start()

    # Create a queue to communicate to the video process.
    videoq = Queue()

    # Create a queue to communicate to the video process.
    streamq = Queue()

    # Create a queue to communicate to the gamepad process.
    gamepadq = Queue()
    # Create a Process for the gamepad, and give it the gamepad queue.
    gamepadp = Process(target=gamepad.gamepadProcess,
                       args=(gamepadq, motorq, videoq, streamq))
    
    # Start the gamepadProcess
    gamepadp.start()

    # Register the exitFunction() to be called when this Python script ends.
    atexit.register(exitFunction)

    # Prevent this main process from terminating until ESCAPE is pressed
    while True:
        time.sleep(1)
    # call the exit function
    exitFunction()
