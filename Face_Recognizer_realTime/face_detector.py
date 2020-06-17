import cv2
import numpy

#location of opencv cascades .xml for face operations
cascade_locaion = r'C:\\Users\\ehsan\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data'
#using front face casacade for detecting the face map
face_cascade = cv2.CascadeClassifier(cascade_locaion+r'\\haarcascade_frontalface_alt2.xml')
#our face recognizer that reads info from face_trainer.py output trained yml file
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("trained_data.yml")

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
        gray_face = gray_pic[y:y+h,x:x+w]
        nbr_predict , loss = face_recognizer.predict(gray_face)
        if loss<68:
            cv2.putText(pic,"ehsan",(x,y-30),cv2.FONT_HERSHEY_PLAIN,4,(0,255,0),2,cv2.LINE_AA)
        else:
            cv2.putText(pic,"not ehsan",(x,y-30),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2,cv2.LINE_AA)
        radious = int(max(w,h)/1.6)
        #making circle over the detected face!
        cv2.circle(pic , (int(x+(w/2)),int(y+(h/2))) , radious , (255,0,255),2)
        
    cv2.imshow("camera",pic)
    #loading frames in every 10 ms using delays, otherwise it doesnt works!
    if cv2.waitKey(10) & 0xFF == ord("0"):
        break
    
camera.release()
cv2.destroyAllWindows()