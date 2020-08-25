import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Lena.png',1)
B,G,R = img[:,:,0],img[:,:,1],img[:,:,2]
plt.hist(B.ravel(),256,[0,256],rwidth=1000); plt.savefig("Blue.png")
plt.hist(G.ravel(),256,[0,256],rwidth=1000); plt.savefig("Green.png")
plt.hist(R.ravel(),256,[0,256],rwidth=1000); plt.savefig("Red.png")