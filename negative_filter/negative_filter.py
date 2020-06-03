import cv2
import numpy

#negative-filter for BGR image
img = cv2.imread("lenna.png" , 1)
cv2.imshow("input" , img)

#getting the width and height for "for" operations
height , width , dim = img.shape

for i in range(height):
    for j in range(width):
        img[i][j] = 255 - img[i][j]

#showing the results        
cv2.imshow("result" , img)
cv2.waitKey(0)