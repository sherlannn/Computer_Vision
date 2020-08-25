import numpy as np
import cv2

#importing and converting to hsv color format
img = cv2.imread('faces.jpeg',1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h = hsv[:,:,0]
s = hsv[:,:,1]
v = hsv[:,:,2]

#tresh-holding image based on skin's saturarion and hue values
ret, min_sat = cv2.threshold(s,40,255, cv2.THRESH_BINARY)
ret, max_hue = cv2.threshold(h,15, 255, cv2.THRESH_BINARY_INV)

#getting final result based on xoring the max-s and min-h values got from previous step
final = cv2.bitwise_and(min_sat,max_hue)
cv2.imshow("Final",final)
cv2.imshow("Original",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
#done