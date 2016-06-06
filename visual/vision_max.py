
import matplotlib.pyplot as plt
import math
import numpy as np
import os
#import cv2
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from scipy.ndimage import convolve

curDir = os.getcwd()
inFile = r'/home/gronerl/ETH/sensys/exercises/visual/Visual/Images/All_Images/lena.jpg'
img = plt.imread(os.path.join(curDir, inFile))
plt.imshow(img)
spread = np.sqrt(img.shape[0]**2+img.shape[1]**2)

print("Please click on the fixation point.")
p = plt.ginput(1)
print("Your fixatoin point has coordinates ", p)

myrangex = np.arange(-p[0][0],img.shape[1]-p[0][0])
myrangey = np.arange(-p[0][1],img.shape[0]-p[0][1])

X,Y = np.meshgrid(myrangex,myrangey)
Z = np.zeros(X.shape)
R = np.sqrt(X**2+Y**2)

def calculate_radius(spread):
    cm_per_pixel = 30/1400
    spread = spread*cm_per_pixel #calculate spread
    spread_retina = math.atan(spread/60)*10 #in mm
    
    stepsize = 0.25 #[mm]
    r = np.arange(0,spread_retina+2*stepsize,stepsize) #distance from fovea [mm] and stepsize [mm] 
    phi = (2*math.pi*r)/(25*math.pi) #calculate angle from distance to fovea
    dist = list()
    for i in range(len(phi)): #calculate distance from fixation point in picture [cm]
        d = math.tan(phi[i])*60
        dist.append(d)
    
    print(dist)
    
    dist_px = list()
    for i in range(len(dist)):  #calculate distance from fixation point in picture [px]
        pix = dist[i]/cm_per_pixel
        dist_px.append(pix)
    
    print(dist_px)
    
    return dist_px

        



dist_px = calculate_radius(spread)

print (len(dist_px))

for i in range(len(dist_px)):
    Z[R>dist_px[i]] = (255/len(dist_px))*i


plt.gray()
plt.imshow(Z)
plt.show()
#circles = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
#print(circles.shape)
#calculate distance from fovea




