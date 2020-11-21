import cv2
import numpy as np
import math
from keras.models import load_model


my_model = load_model('trained_model.h5')

def predict_dig(img):
    test_image = img.reshape(-1,28,28,1)
    return np.argmax(my_model.predict(test_image))

def labeling(img,label,x,y):
    xprime = int(x) - 12
    yprime = int(y) + 12
    cv2.rectangle(img,(xprime,yprime+5),(xprime+35,yprime-35),(255,0,255),-1) 
    cv2.putText(img,str(label),(xprime,yprime), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),1,cv2.LINE_AA)
    return img


def img_inhale(gray_scale):
    org_size = 22
    img_size = 28
    h,w = gray_scale.shape
    
    if h > w:
        factor = org_size/h
        h = org_size
        w = int(round(w*factor))        
    else:
        factor = org_size/w
        w = org_size
        h = int(round(h*factor))
    gray_scale = cv2.resize(gray_scale, (w, h))
    
    wPadding = (int(math.ceil((img_size-w)/2.0)),int(math.floor((img_size-w)/2.0)))
    hPadding = (int(math.ceil((img_size-h)/2.0)),int(math.floor((img_size-h)/2.0)))
    
    gray_scale = np.lib.pad(gray_scale,(hPadding,wPadding),'constant')
    return gray_scale

def extraction(img_location):
    img = cv2.imread(img_location,2)
    img_org =  cv2.imread(img_location)

    ret,threshold = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(threshold, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i,count in enumerate(contours):
        e = 0.01*cv2.arcLength(count,True)
        approx = cv2.approxPolyDP(count,e,True)
        
        hull = cv2.convexHull(count)
        k = cv2.isContourConvex(count)
        x,y,w,h = cv2.boundingRect(count)
        
        if(hierarchy[0][i][3]!=-1 and w>10 and h>10):
            cv2.rectangle(img_org,(x,y),(x+w,y+h),(0,255,0),2)
            region_of_intrest = img[y:y+h, x:x+w]
            region_of_intrest = cv2.bitwise_not(region_of_intrest)
            region_of_intrest = img_inhale(region_of_intrest)
            th,fnl = cv2.threshold(region_of_intrest,127,255,cv2.THRESH_BINARY)
            prediction_out = predict_dig(region_of_intrest)
            print(prediction_out)
            (x,y),radius = cv2.minEnclosingCircle(count)
            img_org = labeling(img_org,prediction_out,x,y)

    return img_org

img = extraction("a.png")	
cv2.imwrite("output.png",img)
cv2.imshow("output",img)
cv2.waitKey(0)