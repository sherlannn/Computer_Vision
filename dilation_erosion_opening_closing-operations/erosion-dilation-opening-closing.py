import cv2
import numpy
 
img = cv2.imread("tst.PNG", 0)
cv2.imshow("original", img)
 
#our kernel
kernel = numpy.ones((5,5), numpy.uint8)
 
#erosion
erosion = cv2.erode(img, kernel, iterations = 1)
cv2.imshow('erosion', erosion)
 
#dilation
dilation = cv2.dilate(img, kernel, iterations = 1)
cv2.imshow('dilation', dilation)
 
#opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
cv2.imshow('opening', opening)
 
#closing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imshow('Closing', closing)

cv2.waitKey(0)