import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import sqrt,exp


img = cv2.imread("lenna.png", 0)

#fourier transform of image
fourier_t = np.fft.fft2(img)
fourier_t_shifted = np.fft.fftshift(fourier_t)


def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

def butterworthLP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base


#ideal lowpass-frequency filter
plt.subplot(3,5,1), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
plt.subplot(3,5,2), plt.imshow(np.log(1+np.abs(fourier_t_shifted)), "gray"), plt.title("shifted fourier t"),plt.yticks([]),plt.xticks([])
ideal_lpf_shifted = fourier_t_shifted * idealFilterLP(60,img.shape)
plt.subplot(3,5,3), plt.imshow(np.log(1+np.abs(ideal_lpf_shifted)), "gray"), plt.title("ideal-lpf fourier shifted"),plt.yticks([]),plt.xticks([])
ideal_lpf = np.fft.ifftshift(ideal_lpf_shifted)
plt.subplot(3,5,4), plt.imshow(np.log(1+np.abs(ideal_lpf)), "gray"), plt.title("ideal-lpf fourier inverse"),plt.yticks([]),plt.xticks([])
ideal_lpf_img = np.fft.ifft2(ideal_lpf)
plt.subplot(3,5,5), plt.imshow(np.abs(ideal_lpf_img), "gray"), plt.title("ideal-lpf image d0=60"),plt.yticks([]),plt.xticks([])

#gaussian lowpass-frequency filter
plt.subplot(3,5,6), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
plt.subplot(3,5,7), plt.imshow(np.log(1+np.abs(fourier_t_shifted)), "gray"), plt.title("shifted fourier t"),plt.yticks([]),plt.xticks([])
gaussian_lpf_shifted = fourier_t_shifted * gaussianLP(60,img.shape)
plt.subplot(3,5,8), plt.imshow(np.log(1+np.abs(gaussian_lpf_shifted)), "gray"), plt.title("gaussian-lpf fourier shifted"),plt.yticks([]),plt.xticks([])
gaussian_lpf = np.fft.ifftshift(gaussian_lpf_shifted)
plt.subplot(3,5,9), plt.imshow(np.log(1+np.abs(gaussian_lpf)), "gray"), plt.title("gaussian-lpf fourier"),plt.yticks([]),plt.xticks([])
gaussian_lpf_img = np.fft.ifft2(gaussian_lpf)
plt.subplot(3,5,10), plt.imshow(np.abs(gaussian_lpf_img), "gray"), plt.title("gaussian-lpf image d0=60"),plt.yticks([]),plt.xticks([])

#butterworth lowpass-frequency filter 
plt.subplot(3,5,11), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
plt.subplot(3,5,12), plt.imshow(np.log(1+np.abs(fourier_t_shifted)), "gray"), plt.title("shifted fourier t"),plt.yticks([]),plt.xticks([])
butterworth_lpf_shifted = fourier_t_shifted * butterworthLP(60,img.shape,3)
plt.subplot(3,5,13), plt.imshow(np.log(1+np.abs(butterworth_lpf_shifted)), "gray"), plt.title("butterworth-lpf fourier shifted"),plt.yticks([]),plt.xticks([])
butterworth_lpf = np.fft.ifftshift(butterworth_lpf_shifted)
plt.subplot(3,5,14), plt.imshow(np.log(1+np.abs(butterworth_lpf)), "gray"), plt.title("butterworth-lpf fourier"),plt.yticks([]),plt.xticks([])
butterworth_lpf_img = np.fft.ifft2(butterworth_lpf)
plt.subplot(3,5,15), plt.imshow(np.abs(butterworth_lpf_img), "gray"), plt.title("butterworth-lpf image n=3 d0=60"),plt.yticks([]),plt.xticks([])


plt.show()
#Ehsan Mokhtari - 1/may/2020