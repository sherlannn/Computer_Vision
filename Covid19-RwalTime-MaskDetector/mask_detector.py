#Author : Ehsan Mokhtari - 2020/08/july

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy
import cv2

#this part is for face detection cascades
cascade_location = r'C:\\Users\\ehsan\\AppData\\Local\\Programs\\Python\\Python38\\Lib\\site-packages\\cv2\\data'
face_cascade = cv2.CascadeClassifier(cascade_location+r'\\haarcascade_frontalface_alt2.xml')

#making instance of our masek_detector trainde model
mask_detector_model = load_model('./mask_detector.h5')

#mask detection function - it returns the probe of mask detected
def mask_detection(face):
    input_frame = face.copy()
    #resizing input image for better performance in processing
    resized = cv2.resize(input_frame, (256, 256))
    resized = img_to_array(resized)
    resized = preprocess_input(resized)
    resized = numpy.expand_dims(resized, axis=0)
    #probe of detected_mask based on out model
    mask_percentage, loss = mask_detector_model.predict([resized])[0]
    return mask_percentage

#running camera(in this case it is webcam) and you can change its mode
camera = cv2.VideoCapture(0)

while True : 
    #val and pic is the output of read() function and pic is the frames of recording webcam
    val,pic = camera.read()
    #making the output to grayscale for better speed in processing
    gray_pic = cv2.cvtColor(pic,cv2.COLOR_BGR2GRAY)
    #detecting the face for probing the detected mask value
    face_detected = face_cascade.detectMultiScale(gray_pic,scaleFactor = 1.1 , minNeighbors = 4)
    #face_detected returns 4 value for the location of every detected face
    for x,y,w,h in face_detected:
        #location of detected face
        face = pic[x:x+w,y:y+h,:]

        try:
            mask_or_not = mask_detection(face)
        
            if mask_or_not>=0.44:
                #detecting faces using circle border
                cv2.rectangle(pic , (x,y),(x+w,y+h) ,(0,255,0),2)
                #mask detected
                cv2.putText(pic,"Mask",(x,y-30),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),1,cv2.LINE_AA)  
            if mask_or_not<0.44:
                #detecting faces using circle border
                cv2.rectangle(pic , (x,y),(x+w,y+h) ,(0,0,255),2)
                #mask not detected
                cv2.putText(pic,"No Mask",(x,y-30),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),1,cv2.LINE_AA)

        except:
            #error handling for not crashing in realtime processing
            pass

    #outputing the realtime results
    cv2.imshow("camera",pic)
    #quit using key "q"
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break
    
camera.release()
cv2.destroyAllWindows()
#done