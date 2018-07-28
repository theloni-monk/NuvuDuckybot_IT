from rpistream.camera import camera
from rpistream.streamserver import Server

def retrieveImage(cam,imgResize):
    image = cv2.resize(cam.image,(0,0),fx=imgResize,fy=imgResize)
    return image

cam = rpistream.camera.Camera(mirror=True)
scale=0.5
server = Server(port=5000)
server.serve() # Blocking; waits for a connection before continuing
server.startStream(retrieveImage,[cam,scale]) # Calls retrieveImage(*args) every frame  