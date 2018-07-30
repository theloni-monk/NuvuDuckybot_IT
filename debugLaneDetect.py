from laneDetection2 import *
from rpistream import *

def makeImg(cam, dF, scale):
    image = cv2.resize(cam.image,(0,0),fx=scale,fy=scale)
    return (dF.process(res)*255)#.astype("uint8") detectFunc(image)

Ld= LaneDetector() #needs more params
cam=camera.Camera()
scale=1
p=ColorProfile.lanes
calibImg = Ld.getCalibImage(cam)
res=Ld.calibrate(calibImg, p, debug=True)

server = streamserver.Server(port=5000)
server.serve() # Blocking; waits for a connection before continuing
server.startStream(makeImg,[cam, Ld, scale]) # Calls retrieveImage(*args) every frame  
