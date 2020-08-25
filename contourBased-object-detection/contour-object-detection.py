#Ehsan Mokhtari
import numpy
import cv2
import random

#importing input image and converting it to gray scale for tresh-holding process
img = cv2.imread('input.png',1)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#adaptive-tresh-holding for making perfect binary-image for object detection
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 141, 1)
#cv2.imshow("Binary", thresh)

#extracting contours from binary-image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#making a dark 3-channel panel for drawing contours with different colors on it 
panel = numpy.zeros([img.shape[0], img.shape[1],3], 'uint8')

#counter for objects
counter=1

for c in contours:
	#making random color for objs
	color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
	#drawing and filling the contours on panel
	cv2.drawContours(panel, [c], -1, color, -1)

	#area of detected obj
	area = cv2.contourArea(c)

	#calculating center of every object for putting its number on it!
	M = cv2.moments(c)
	cx = int( M['m10']/M['m00'])
	cy = int( M['m01']/M['m00'])

	#put the number of obj on it
	cv2.putText(panel,str(counter),(cx-15,cy+10),cv2.FONT_ITALIC,1,(0,0,0),1,cv2.LINE_AA)
	
	#outputing the results
	print("object: {}, Area: {}, location (x={}, y={})".format(counter,area,cx,cy))

	#counter increases by 1
	counter = counter+1

cv2.imshow("Contours",panel)

cv2.waitKey(0)
cv2.destroyAllWindows()
#End