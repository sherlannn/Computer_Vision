import cv2
import numpy

#location of opencv cascades .xml for face operations
cascade_locaion = r'C:\\Users\\ehsan\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data'
#using front face casacade for detecting the face map
face_cascade = cv2.CascadeClassifier(cascade_locaion+r'\\haarcascade_frontalface_alt2.xml')

camera = cv2.VideoCapture(0)

while True:
    #it returns successfull or not value and picture values
    val,pic = camera.read()

    #cascade works in gray scale for now so we have to make our input pic in gray lvl
    gray_pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    #scaleFactor = how the size of captured image reduced 1.05 good detectio & 1.5 faster.its your choise!
    #minNeighbor = how many neighbors each candidate rectangle should have to retain it 3~6 is good for it
    face_detected = face_cascade.detectMultiScale(gray_pic,scaleFactor = 1.1 , minNeighbors = 4)
    for x,y,w,h in face_detected:
        radious = int(max(w,h)/1.6)
        #making circle over the detected face!
        cv2.circle(pic , (int(x+(w/2)),int(y+(h/2))) , radious , (255,0,255),2)
        
    cv2.imshow("camera",pic)
    #loading frames in every 10 ms using delays, otherwise it doesnt works!
    if cv2.waitKey(10) & 0xFF == ord("0"):
        break
    
camera.release()
cv2.destroyAllWindows()