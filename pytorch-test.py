# Python code for Multiple Color Detection 


import numpy as np 
import cv2 
import torchvision.transforms as transforms

# Capturing video through webcam 
webcam = cv2.VideoCapture(0) 

# Start a while loop 
while(1): 
	
	# Reading the video from the 
	# webcam in image frames 
	_, imageFrame = webcam.read() 
	
	imageFrame2 = imageFrame
	imageFrame2 = cv2.cvtColor(imageFrame2, cv2.COLOR_BGR2RGB)
	tensorImg = transforms.ToTensor()(imageFrame2)

	# Program Termination 
	cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
	if cv2.waitKey(10) & 0xFF == ord('q'): 
		webcam.release() 
		cv2.destroyAllWindows() 
		break
