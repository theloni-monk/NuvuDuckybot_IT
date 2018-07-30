from laneDetection2 import *
from rpistream import *

def makeImg(cam, detectFunc, scale, args):
    image = cv2.resize(cam.image,(0,0),fx=scale,fy=scale)
    return detectFunc(image,*args)

Ld= LaneDetector() #needs more params
cam=camera.Camera()
scale=1
server = streamserver.Server(port=5000)
server.serve() # Blocking; waits for a connection before continuing
server.startStream(makeImg,[cam, Ld.calibrate, scale,[ColorProfile.lanes]]) # Calls retrieveImage(*args) every frame  