import cv2
import numpy

img = cv2.imread("test.jpg",0)
cv2.imshow("input" , img)

out = cv2.equalizeHist(img)
cv2.imshow("result" , out)

cv2.waitKey(0)