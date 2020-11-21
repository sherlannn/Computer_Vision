#Ehsan Mokhtari
import cv2
import numpy

#haarcascade for eye detection
eye_cascade_path  = "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)


#opening webcam
capture  = cv2.VideoCapture(0)

while True:
	
	#capture.read() returns two obj.the one we need is frame one!
    return_ , frame = capture.read()
	#converting to gray scale for speed optimization
    gray_scaled = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	#detecting eye posiotions from gray scaled frames
    eyes = eye_cascade.detectMultiScale(gray_scaled, scaleFactor=1.02,minNeighbors=20,minSize=(10,10))

	#for each eye position draw rectangle around it
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0))

	
    cv2.imshow("output",frame)
    ch = cv2.waitKey(10)
    if ch & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()