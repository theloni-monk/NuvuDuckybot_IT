import cv2
from camera import Camera
import numpy as np
import time
def grayscale(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def autoCanny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def unzero(x):
    if x==0:
        x = 0.001
    return x
def getLineColor(img,m,b):
    
def process(color):
    color = color[color.shape[0]//2:,:]
    img = grayscale(color)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)
    edges = autoCanny(img)
    output = color*edges.reshape([edges.shape[0],edges.shape[1],1])

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    if lines is None:
        return output
    for line in lines:
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
        
            m = (y2-y1)/(unzero(x2-x1))
            b = y1-m*x1

            cv2.line(output,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(output,(int(x0),int(y0)),4,(255,0,0),-1)
    return output

if __name__ == "__main__":
    cam = Camera(mirror=True)
    time.sleep(5)
    cv2.imwrite("test/lanes.png",process(cam.image))
    raise SystemExit
    while 1:
        cv2.imshow('my webcam', process(cam.image))
        if cv2.waitKey(1) == 27:
            break  # esc to quit
