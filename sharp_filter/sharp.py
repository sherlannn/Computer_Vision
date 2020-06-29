import cv2
import numpy

img = cv2.imread("lenna.png",0)

#make sharp
sharp_filter = numpy.array(([-1,-1,-1],
                            [-1,9,-1],
                            [-1,-1,-1]))
sharp = cv2.filter2D(img,-1,sharp_filter)

cv2.imshow("sharp",sharp)
cv2.waitKey(0)