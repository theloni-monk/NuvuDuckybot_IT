import laneDetection
import PID.PID
import rpistream.camera.Camera

p,i,d=0.5,0.05,0.01
pid=PID(p,i,d)

ld=LaneDetector()

#PID off of the pprocess4 output
#convert output to diff vector \<-I 
#norm that vector and send to motorq
