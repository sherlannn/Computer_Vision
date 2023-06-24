import cv2

img = cv2.imread("lenna.png" , 1)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
inverGray = 255 - imgGray
inverGray = cv2.GaussianBlur(inverGray , (23,23),0)
out = cv2.divide(imgGray , 255 - inverGray , scale= 256.0)

cv2.imshow("output" , out)
cv2.waitKey(0)