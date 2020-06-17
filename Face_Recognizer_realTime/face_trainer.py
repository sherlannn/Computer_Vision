import os
import cv2
import numpy

#pip install opencv-contrib-python
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

cascade_locaion = r'C:\\Users\\ehsan\\AppData\\Local\\Programs\\Python\\Python38-32\\Lib\\site-packages\\cv2\\data'
face_cascade = cv2.CascadeClassifier(cascade_locaion+r'\\haarcascade_frontalface_alt2.xml')

img_dr = r'C:\\Users\\ehsan\\Documents\\Python projects\\image face 2\\images'
img_ls = os.listdir(img_dr)

#i use just my face to train so the y_train is just ^ehsan^ lable
x_train = []

for i in range(len(img_ls)):
    img_lc = img_dr+r'\\'+img_ls[i] 
    print(img_lc)
    #for reducing the unnecessary computing, we imported the images in gray scale
    img = cv2.imread(img_lc,0)
    img_onlyface = face_cascade.detectMultiScale(img,scaleFactor = 1.1 , minNeighbors = 4)
    for x,y,w,h in img_onlyface:
        fc = img[y:y+h,x:x+w]
    x_train.append(fc)

y_train =[0] * len(x_train)
face_recognizer.train(x_train,numpy.array(y_train))
face_recognizer.save("trained_data.yml")