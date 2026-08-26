import cv2
import numpy as np
image = cv2.imread('2.png')
laneimage= np.copy(image)
gray=cv2.cvtColor(laneimage, cv2.COLOR_BGR2GRAY)
cv2.imshow('result',gray)
cv2.waitKey(0) 

