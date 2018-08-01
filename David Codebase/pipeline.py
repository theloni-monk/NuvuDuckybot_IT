import cv2

# The pipeline function takes in a numpy array of dimensions:
#	"height, width, color-space" 
# and MUST return an image of the SAME dimensions
#
# The pipeline function also takes a motorq. To make the motors move
# add messages to the queue of the form:
#	motorq.put( [ left-motor-speed , right-motor-speed ] )
# i.e.	motorq.put([32768,32768]) # make the motors go full-speed forward
def pipeline(image,motorq):
	print("running pipeline...")
		
	# THINGS YOU SHOULD DO...
	# 1. Copy the code INSIDE your pipeline function here.
	# 2. Ensure the pipeline function takes BOTH the image and motorq.

	#motorq.put([32768,32768]) # make the motors go full-speed forward



	return image
