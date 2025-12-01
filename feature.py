import cv2  
  
# np is an alias pointing to numpy library 
import numpy as np 


frame = cv2.imread("static/trained/train1.jpg")
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
      
# define range of red color in HSV 
lower_red = np.array([30,150,50]) 
upper_red = np.array([255,255,180]) 
  
# create a red HSV colour boundary and  
# threshold HSV image 
mask = cv2.inRange(hsv, lower_red, upper_red) 

# Bitwise-AND mask and original image 
res = cv2.bitwise_and(frame,frame, mask= mask)

edges = cv2.Canny(frame,100,200)
cv2.imshow('Edges',edges) 
