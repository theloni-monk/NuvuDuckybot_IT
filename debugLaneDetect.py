import laneDetection
import laneDetection2
import rpistream







def makeImg(cam, detectFunc, scale, args):
    image = cv2.resize(cam.image,(0,0),fx=scale,fy=scale)
    return detectFunc(image)


Ld=laneDetector() #needs more params
cam=rpistream.camera.Camera()
scale=1
server = rpistream.streamserver.Server(port=5000)
server.serve() # Blocking; waits for a connection before continuing
server.startStream(makeImg,[cam,makeImg,scale]) # Calls retrieveImage(*args) every frame  