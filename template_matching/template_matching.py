#Ehsan Mokhtari
import numpy
import cv2

#Reading input pattern and frame images
pattern = cv2.imread('pattern.jpg',0)
frame = cv2.imread("frame.jpg",1)
frame_grayed = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
a,b=pattern.shape

#template matching result
result = cv2.matchTemplate(frame_grayed, pattern, cv2.TM_CCOEFF_NORMED)

#extracting information about max and min values of grayed-scaled result with locations
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_val,max_loc)

#circling around max-value location (the best matched one)
cv2.rectangle(frame,max_loc,(max_loc[0]+b,max_loc[1]+a),(255,255,255),2)
#cv2.imwrite("result.png",frame)
cv2.imshow("result.png",frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
#End