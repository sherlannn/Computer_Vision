import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.util import random_noise

#reading input image     "your image location"
input_image = cv2.imread('lenna.png', 1)
#turning br format to rgb for showing results in right format on matplotlib
input_image = cv2.cvtColor(input_image,cv2.COLOR_BGR2RGB)

#making salt and pepper noise on image using skimage library
#mode = s&p/gaussian/speckle
noisy_image = random_noise(input_image, mode='s&p', seed=None, clip=True)
noisy_image = noisy_image.astype('float32')

#noise smotthing using median smoothing on opencv
smoothed_image = cv2.medianBlur(noisy_image,3)

#ploting the results
plt.subplot(131), plt.imshow(input_image), plt.title('input img'),plt.yticks([]),plt.xticks([])
plt.subplot(132), plt.imshow(noisy_image), plt.title('Sp noise'),plt.yticks([]),plt.xticks([])
plt.subplot(133), plt.imshow(smoothed_image), plt.title('median smoothing'),plt.yticks([]),plt.xticks([])

plt.show()

#cause values are between 0 and 1, for saving you should *255
#cv2.imwrite("output.png",smoothed_image*255)
