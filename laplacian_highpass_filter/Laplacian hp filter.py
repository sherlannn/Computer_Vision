import cv2
import numpy as np

#the input type is BGR but cv2.imshow works well with it
img = cv2.imread("lenna.png",1)
cv2.imshow("original", img)

#grayscaling image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray scale", img)

#laplacian filter
img = cv2.Laplacian(img, cv2.CV_16S)
img =cv2.convertScaleAbs(img)
cv2.imshow("finally", img)

cv2.waitKey(0)