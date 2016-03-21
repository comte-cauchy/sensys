import thLib
import numpy as np
import matplotlib.pyplot as plt
inSound = thLib.sounds.Sound(inFile='SoundData/tiger.wav')
data = .5*inSound.data[:,0]+.5*inSound.data[:,1]
sampleRate = 1000
freq = [50, 120]
amp = [1, 0.4]
t = np.arange(0,10,1./sampleRate)
data =  amp[0]*np.sin(2*np.pi*freq[0]*t) + amp[1]*np.sin(2*np.pi*freq[1]*t) + np.random.randn(len(t))
print(type(inSound.data))
#plt.plot(data) #np.fft.fft(inSound.data)
transformedData = np.fft.fft(data)
plt.plot(np.abs(transformedData))
plt.show()