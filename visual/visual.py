import matplotlib.pyplot as plt
import os
import numpy as np
import skimage as ski
import scipy.signal as spsig
from skimage.color import rgb2gray


def gaborKernel():
    Lambda = 8 #freq sin
    theta = np.pi/2 #orientation (0=horinzontal)
    psi = np.pi/2 #phase-shift
    sigma = 3 #size of gaussian
    gamma = 1 #aspect ratio
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb    
    

def dogKernel(eccentricity,ppmm):
    sqr2pi = np.sqrt(2*np.pi)
    
    r =  eccentricity*10
    sig1 = r/5
    sig2 = sig1*1.6
    
    m = int(r*ppmm)
    if m==0:
        return np.ndarray(shape=(3,3),buffer=np.array([0,0,0,0,1.,0,0,0,0]))
    
    n = 2*m+1
    
    x2 = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            x2[i,j]=(r*(m-i)/m)**2+(r*(m-j)/m)**2
    return (1/(sig1*sqr2pi)**2)*np.exp(-x2/(2*sig1*sig1)) - (1/(sig2*sqr2pi)**2)*np.exp(-x2/(2*sig2*sig2))
            
def getZones(img, p, eccentPP, nZones):
    maxX = np.max([p[0],img.shape[0]-p[0]])
    maxY = np.max([p[1],img.shape[1]-p[1]])
    maxDist = np.sqrt(maxX**2+maxY**2)
    
    radii = np.linspace(0,maxDist,nZones+1)[:-1]
    distances2 = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            distances2[i,j]=(p[0]-i)**2+(p[1]-j)**2
    Z = np.zeros(img.shape)
    for i in range(len(radii)):
        Z[distances2>=(radii[i]**2)] = i
        
    maxEccentricity = maxDist*eccentPP
    eccentricities = np.linspace(0,maxEccentricity,nZones+1)[1:]
    return [Z, eccentricities]



def main():

    nZones = 10
    
    ppmm = 1400/300         # [px/mm]
    screenDistance = 600    # [mm]
    rEye = 12.5             # [mm]
    mmppRetina = (1/ppmm)*rEye/screenDistance # [mm/px]: eccentricity in mm per pixel of the screen
    
    
    curDir = r'/home/gronerl/ETH/sensys/exercises/visual/Visual/Images/All_Images'
    inFile = r'lena_bw.jpg'
    img = plt.imread(os.path.join(curDir, inFile))
    plt.title('Please click on the fixation point.')
    if len(img.shape)==3:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap='gray')        
    
    print("Please click on the fixation point.")
    p = plt.ginput(1)[0]
    plt.close()
    p = (p[1],p[0])
    print("Your fixation point has coordinates ", p)

    img = rgb2gray(img)
    

    [Z,eccentricities] = getZones(img,p ,mmppRetina,nZones)
    
    dogImg = np.zeros(img.shape)
    for i in range(len(eccentricities)):
        kernel = dogKernel(eccentricities[i],ppmm)
        print('Convolution for zone ',i+1, 'of',nZones,'. Kernel has shape ',kernel.shape, 'ecc=',eccentricities[i])
        convImg = spsig.fftconvolve(img,kernel, mode='same')
        convImg = convImg-np.min(convImg)
        convImg = convImg/np.max(convImg)
        dogImg[Z==i]=convImg[Z==i]
    
    #outimg = outimg-np.min(outimg)
  # outimg = outimg/np.max(outimg)
    plt.imshow(dogImg,cmap='gray')
    plt.show()
    
    gaborImg = spsig.fftconvolve(img,gaborKernel(),mode='same')
    plt.imshow(gaborImg,cmap='gray')
    plt.show()
    
if __name__=='__main__':
    main()