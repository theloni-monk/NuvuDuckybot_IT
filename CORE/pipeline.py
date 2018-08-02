import cv2
from laneDetection import *
from PID import *
import time

# The pipeline function takes in a numpy array of dimensions:
#	"height, width, color-space"
# and MUST return an image of the SAME dimensions
#
# The pipeline function also takes a motorq. To make the motors move
# add messages to the queue of the form:
#	motorq.put( [ left-motor-speed , right-motor-speed ] )
# i.e.	motorq.put([32768,32768]) # make the motors go full-speed forward
usePid=True
maxSpeed=32767 #this the default speed
rotConstant=7.9135    #rads/sec


p, i, d = -0.5, 0.05, 0.01
pid = PID(p, i, d)


def normVect(v):
	mag = np.sqrt(np.power(v[0], 2)+np.power(v[1], 2))
	return (v[0]/mag, v[1]/mag)


startT = 0
# dummy initial values
prev = (300, 50, 175)


def pipeline(image, motorq, ld, img=False):
	print("running pipeline...")

	# THINGS YOU SHOULD DO...
	# 1. Copy the code INSIDE your pipeline function here.
	# 2. Ensure the pipeline function takes BOTH the image and motorq.

	# motorq.put([32768,32768]) # make the motors go full-speed forward

	speed = maxSpeed  # TBD

	params = None
	if img:
		various = ld.process4(image, imgOut=True)
		imgO = various[1]
		params = various[0]
	else:
		params=ld.process4(image)

	Re=params[0][0] #road edge
	Rc=params[0][1] #road center
	if Re==None:
		Re=prev[0]
	if Rc==None:
		Rc=prev[1]

	Lc=(Re+Rc)/2 #lane center

	prev=(Re,Rc,Lc)
	RBpos=image.shape/2 #robot position

	# averageing to reduce noise:
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
	diff=(np.atan2(speedVect[0],speedVect[1])/(t*rotConstant))
	start=time.time()

	speed-=diff
	motorq.put([speed-diff,speed+diff]) #speed will never be actual speed


	if img:
		return imgO
