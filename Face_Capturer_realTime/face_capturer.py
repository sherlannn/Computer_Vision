import time
import cv2
import numpy


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
        img = pic[y:y+h,x:x+w,:]
        cv2.imwrite(str(time.time())+".png",img)

    cv2.imshow("camera",pic)
    #loading frames in every 1s using delays, otherwise it doesnt works!
    if cv2.waitKey(1000) & 0xFF == ord("0"):
        break
    
camera.release()
cv2.destroyAllWindows()

