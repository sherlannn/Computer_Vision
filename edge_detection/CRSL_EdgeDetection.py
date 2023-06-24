#Ehsan Mokhtari
import cv2
import numpy

img = cv2.imread("lenna.png",0)

#canny
canny = cv2.Canny(img,100,200)

#laplacian edge detection
#laplacian = cv2.Laplacian(img,cv2.CV_64F,ksize=3)
laplacian_filter = numpy.array(([0,-1,0],
                                [-1,4,-1],
                                [0,-1,0]))
laplacian = cv2.filter2D(img,-1,laplacian_filter)

#sobel
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobelx_filter = numpy.array(([-1,0,1],
                             [-2,0,2],
                             [-1,0,1]))
sobely_filter = numpy.array(([1,2,1],
                             [0,0,0],
                             [-1,-2,-1]))
sobelx = cv2.filter2D(img,-1,sobelx_filter)
sobely = cv2.filter2D(img,-1,sobely_filter)

#robert
roberts_x = numpy.array(([0,0,0],
                         [0,1,0],
                         [0,0,-1]) )
roberts_y = numpy.array(([0,0,0],
                         [0,0,1],
                         [0,-1,0]))
roberty = cv2.filter2D(img,-1,roberts_y)
robertx = cv2.filter2D(img,-1,roberts_x)


cv2.imshow("canny",canny)
cv2.imshow("laplacian",laplacian)
cv2.imshow("sobelx",sobelx)
cv2.imshow("sobely",sobely)
cv2.imshow("robertx",robertx)
cv2.imshow("roberty",roberty)
cv2.waitKey(0)