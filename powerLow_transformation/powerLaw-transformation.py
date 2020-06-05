import cv2
import numpy

img = cv2.imread("lenna.png" , 0)
cv2.imshow("input image" , img)
img = img/255

#getting the width and height for "for" operations
height , width = img.shape

c = 1
y=[4,2,.8,.5,.3]
out  = img.copy()
for n in range(len(y)):
    for i in range(height):
        for j in range(width):
            out[i][j] = c * img[i][j] ** y[n]
    cv2.imshow("Y = {}".format(y[n]) , out)

cv2.waitKey(0)