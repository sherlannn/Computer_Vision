import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('test.jpg',0)
#arrenging pixel values between 0 and 1 for log level computation
img = img/255  
cv2.imshow("input img",img)

rows,cols=img.shape

rh, rl, cutoff = 2.5,0.5,8

#log of the image
img_log = np.log(img+.01)
#fourier transform of image
fourier_log = np.fft.fft2(img_log)
#centering the fourier transform
fourier_log_centered = np.fft.fftshift(fourier_log)

#homomorphic_filter
DX = cols/cutoff
homomorphic_filter = np.ones((rows,cols))
for i in range(rows):
    for j in range(cols):
        homomorphic_filter[i][j]=((rh-rl)*(1-np.exp(-((i-rows/2)**2+(j-cols/2)**2)/(2*DX**2))))+rl

#making the homomorphic filter on fourier transformed image
result_homomorphic_fourier_img = homomorphic_filter * fourier_log_centered
#reverse fourier transform
result_homomorphic_log_img = np.real(np.fft.ifft2(np.fft.ifftshift(result_homomorphic_fourier_img)))
#exp of the reversed fourier that is logaritmed before
result = np.exp(result_homomorphic_log_img)

#showing the output
cv2.imshow("result",result)
cv2.waitKey(0)