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

def idealFilterHP(D0,imgShape):
    base = np.ones(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 0
    return base

def butterworthHP(D0,imgShape,n):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1-1/(1+(distance((y,x),center)/D0)**(2*n))
    return base

def gaussianHP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = 1 - exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

#ideal lowpass-frequency filter
plt.subplot(3,4,1), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
ideal_hpf_shifted = fourier_t_shifted * idealFilterHP(60,img.shape)
plt.subplot(3,4,2), plt.imshow(np.log(1+np.abs(ideal_hpf_shifted)), "gray"), plt.title("ideal-hpf fourier shifted"),plt.yticks([]),plt.xticks([])
ideal_hpf = np.fft.ifftshift(ideal_hpf_shifted)
plt.subplot(3,4,3), plt.imshow(np.log(1+np.abs(ideal_hpf)), "gray"), plt.title("ideal-hpf fourier"),plt.yticks([]),plt.xticks([])
ideal_hpf_img = np.fft.ifft2(ideal_hpf)
plt.subplot(3,4,4), plt.imshow(np.log(1+np.abs(ideal_hpf_img)), "gray"), plt.title("ideal-hpf image d0=60"),plt.yticks([]),plt.xticks([])

#gaussian lowpass-frequency filter
plt.subplot(3,4,5), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
gaussian_hpf_shifted = fourier_t_shifted * gaussianHP(60,img.shape)
plt.subplot(3,4,6), plt.imshow(np.log(1+np.abs(gaussian_hpf_shifted)), "gray"), plt.title("gaussian-hpf fourier shifted"),plt.yticks([]),plt.xticks([])
gaussian_hpf = np.fft.ifftshift(gaussian_hpf_shifted)
plt.subplot(3,4,7), plt.imshow(np.log(1+np.abs(gaussian_hpf)), "gray"), plt.title("gaussian-hpf fourier"),plt.yticks([]),plt.xticks([])
gaussian_hpf_img = np.fft.ifft2(gaussian_hpf)
plt.subplot(3,4,8), plt.imshow(np.log(1+np.abs(gaussian_hpf_img)), "gray"), plt.title("gaussian-hpf image d0=60"),plt.yticks([]),plt.xticks([])

#butterworth lowpass-frequency filter 
plt.subplot(3,4,9), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
butterworth_hpf_shifted = fourier_t_shifted * butterworthHP(60,img.shape,3)
plt.subplot(3,4,10), plt.imshow(np.log(1+np.abs(butterworth_hpf_shifted)), "gray"), plt.title("butterworth-hpf fourier shifted"),plt.yticks([]),plt.xticks([])
butterworth_hpf = np.fft.ifftshift(butterworth_hpf_shifted)
plt.subplot(3,4,11), plt.imshow(np.log(1+np.abs(butterworth_hpf)), "gray"), plt.title("butterworth-hpf fourier"),plt.yticks([]),plt.xticks([])
butterworth_hpf_img = np.fft.ifft2(butterworth_hpf)
plt.subplot(3,4,12), plt.imshow(np.log(1+np.abs(butterworth_hpf_img)), "gray"), plt.title("butterworth-hpf image n=3 d0=60"),plt.yticks([]),plt.xticks([])


plt.show()
#Ehsan Mokhtari - 1/may/2020