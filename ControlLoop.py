import laneDetection
from PID import *
import rpistream.camera.Camera
import time

usePid=True
maxSpeed=32767 #this the default speed
rotConstant=7.9135    #rads/sec

cam=Camera()
ld=LaneDetector()
ld.calibrateKmeans(cv2.imread("calib.png"))

p,i,d= -0.5, 0.05, 0.01
pid=PID(p,i,d)


def normVect(v):
    mag= np.sqrt(np.power(v[0],2)+np.power(v[1],2))
    return (v[0]/mag,v[1]/mag)

startT=0
#dummy initial values
prev=(300,50,175)

while True:
    speed=maxSpeed #TBD
    
    img=cam.image
    params=ld.process4(img)
    Re=params[0][0] #road edge
    Rc=params[0][1] #road center
    if Re==None:
        Re=prev[0]
    if Rc==None:
        Rc=prev[1]

    Lc=(Re+Rc)/2 #lane center

    prev=(Re,Rc,Lc)
    RBpos=img.shape/2 #robot position

    #averageing to reduce noise:
    Re,Rc,Lc=ld.rollingAverage(Re),ld.rollingAverage(Rc),ld.rollingAverage(Lc)
    
    Cdiff=RBpos-Lc

    outputdiff=-Cdiff
    if usePid:
        pid.setSetpoint(0)
        outputdiff=pid.update(Cdiff)
    
    outputVect=normVect((outputdiff,1))

    speedVect=(outputVect[0]*speed,outputVect[1]*speed)
    endT=time.time
    t=endT-startT
    diff=(np.atan2(x,y)/(t*rotConstant))
    start=time.time()

    speed-=diff
    motorq.put([speed-diff,speed+diff]) #speed will never be actual speed


#PID off of the pprocess4 output
#convert output to diff vector \<-I 
#norm that vector and send to motorq
