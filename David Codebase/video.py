
import time
import cv2
import os
# Import library that allows parallel processing
from multiprocessing import Process, Queue
# Import library for streaming video
from rpistream.streamserver import Server
# Import the pipeline code
import pipeline
# Import the debug constant
from debug import VERBOSE
import socket

# Change the camera resolution, before the processes start
cam_width = 320
cam_height = 240

def retrieveImage(cam,motorq):
	# read a frame from the camera
	ret, frame = cam.read()
	if not ret:
		# return a black frame when the camera retrieves no frame
		return np.zeros_like(frame)
	# try calling the pipeline function
	frame = pipeline.pipeline(frame,motorq)
	return frame


def streamProcess(motorq,streamq):
	global cam_width, cam_height, scale
	server = Server(port=5000,verbose=VERBOSE)
	disconnected = True
	cam = cv2.VideoCapture(0)
	cam.set(3,cam_width)
	cam.set(4,cam_height)
	while True:
		# we are now in the video loop, check if we should exit
		msg = None
		# Get the most recent message
		while not streamq.empty():
			msg = streamq.get(block=False)
		# Check if the message is None or "exit"
		if msg is None:
			pass
		elif msg == 'exit':
			return
		
		try:
			if disconnected:
				server.serveNoWait()
			disconnected = False
			server.stream(retrieveImage,[cam,motorq])
		except socket.error as exc:
			print (exc)
			disconnected = True

	# release the camera
	cam.release()


def videoProcess(motorq,videoq):
	global cam_width, cam_height
	cam = cv2.VideoCapture(0)
	cam.set(3,cam_width)
	cam.set(4,cam_height)
	# spwan the streaming video process

	while True:
		# we are now in the video loop, check if we should exit
		msg = None
		# Get the most recent message
		while not videoq.empty():
			msg = videoq.get(block=False)
		# Check if the message is None or "exit"
		if msg == None:
			pass
		elif msg == "exit":
			# Quit this function if the message is None
			# This is the indicator to stop this function
			return
			
		# read a frame from the camera
		ret, frame = cam.read()
		if not ret:
			# return a black frame when the camera retrieves no frame
			return

		# try calling the pipeline function
		frame = pipeline.pipeline(frame,motorq)

		#cv2.imshow("test", frame)
		#k = cv2.waitKey(1)
	
	# release the camera
	cam.release()








