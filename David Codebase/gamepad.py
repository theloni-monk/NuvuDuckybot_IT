#!/usr/bin/python
# Import the device reading library
from evdev import InputDevice, categorize, ecodes, KeyEvent, list_devices
import video
# Import library that allows parallel processing
from multiprocessing import Process, Queue


# Get the name of the Logitech Device
def getInputDeviceByName(name):
    devices = [InputDevice(fn) for fn in list_devices()]
    for device in devices:
        if device.name == name:
            return InputDevice(device.fn)
    return None


# Import our gamepad.
gamepad = getInputDeviceByName('Logitech Gamepad F710')

# Keeps track of which Video Process is running
videoRunning = False
streamRunning = False


def clearQueue(q):
    while not q.empty():
        q.get(block=False)

# Process the GamePad


def gamepadProcess(gamepadq, motorq, videoq, streamq):
    # Create variables to keep track of the joystick state.
    joyLR = 0
    joyUD = 0
    # Create variable to keep track of camera state
    global videoRunning, streamRunning
    # Loop over the gamepad's inputs, reading it.
    for event in gamepad.read_loop():
        # we are now in the gamepad loop, check if we should exit
        msg = None
        # Get the most recent message
        while not gamepadq.empty():
            msg = gamepadq.get(block=False)
        # Check if the message is None or "exit"
        if msg is None:
            pass
        if msg == "exit":
            # Quit this function if the message is None
            # This is the indicator to stop this function
            return

        # continue processing gamepad values
        if event.type == ecodes.EV_KEY:
            keyevent = categorize(event)
            if keyevent.keystate == KeyEvent.key_down:
                print(keyevent.keycode)
                # example key detection code
                if 'BTN_TL' in keyevent.keycode:
                    if videoRunning:
                        print("DISABLE PIPELINE")
                        # Turn the camera OFF
                        videoRunning = False
                        videoq.put('exit')
                        videop.join()
                    else:
                        print("ENABLE PIPELINE")
                        if streamRunning:
                            streamRunning = False
                            streamq.put('exit')
                            streamp.join()
                        clearQueue(videoq)
                        # Turn the camera ON
                        videoRunning = True
                        # Create a Process for the camera, and give it the video queue.
                        videop = Process(
                            target=video.videoProcess, args=(motorq, videoq))
                        # Start the videoProcess
                        videop.start()

                if 'BTN_TR' in keyevent.keycode:
                    if streamRunning:
                        print("DISABLE STREAMING VIDEO & PIPELINE")
                        # Turn the camera OFF
                        streamRunning = False
                        streamq.put('exit')
                        streamp.join()
                    else:
                        print("ENABLE STREAMING VIDEO & PIPELINE")
                        if videoRunning:
                            videoRunning = False
                            videoq.put('exit')
                            videop.join()
                        clearQueue(streamq)
                        # Turn the camera ON
                        streamRunning = True
                        # Create a Process for the camera, and give it the video queue.
                        streamp = Process(
                            target=video.streamProcess, args=(motorq, streamq))
                        # Start the videoProcess
                        streamp.start()

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
                joyLR = event.value
                # Send a message to the motorProcess when the joystick moves.
                motorq.put([joyUD+joyLR, joyUD-joyLR])
            elif event.code == 4:
                print('JOY_UD '+str(event.value))
                joyUD = event.value
                # Send a message to the motorProcess when the joystick moves.
                motorq.put([joyUD+joyLR, joyUD-joyLR])
            elif event.code == 5:
                print('TRIG_R '+str(event.value))
            elif event.code == 16:
                print('HAT_LR '+str(event.value))
            elif event.code == 17:
                print('HAT_UD '+str(event.value))
            else:
                pass
