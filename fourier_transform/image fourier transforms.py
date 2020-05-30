import cv2
import numpy as np
from matplotlib import pyplot as plt

#reading image in gray scale mode
img = cv2.imread("lenna.png", 0)

#fast fourier transform
fourier_t = np.fft.fft2(img)

#centeralizing the fourier transform
fourier_t_shifted = np.fft.fftshift(fourier_t)

#decentralizing the shifted fourier transform
reverse_fourier_shifted = np.fft.ifftshift(fourier_t_shifted)

#inversing the fourier transform to real image
reverse_real_image = np.fft.ifft2(reverse_fourier_shifted)

#ploting
plt.subplot(151), plt.imshow(img, "gray"), plt.title("original image"),plt.yticks([]),plt.xticks([])
plt.subplot(152), plt.imshow(np.log(1+np.abs(fourier_t)), "gray"), plt.title("fourier t"),plt.yticks([]),plt.xticks([])
plt.subplot(153), plt.imshow(np.log(1+np.abs(fourier_t_shifted)), "gray"), plt.title("fourier t shifted"),plt.yticks([]),plt.xticks([])
plt.subplot(154), plt.imshow(np.log(1+np.abs(reverse_fourier_shifted)), "gray"), plt.title("reverse fs"),plt.yticks([]),plt.xticks([])
plt.subplot(155), plt.imshow(np.abs(reverse_real_image), "gray"), plt.title("reversed original image"),plt.yticks([]),plt.xticks([])

plt.show()