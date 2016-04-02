import numpy as np
import matplotlib.pyplot as plt
import GammaTones as gt

nEl = 21
sampleRate = 44100 #hz = 1/s
windowTime = 6 #ms
wSize = windowTime*0.001*sampleRate
t = np.linspace(0,5,5*sampleRate)
data = np.sin(2*np.pi*440*t)
window = np.zeros(data.shape)
window[0:wSize] = 1.

(forw, feed, cf, ERB, B) = gt.GammaToneMake(sampleRate, nEl, 200, 5000, 'moore')


window = np.zeros(data.shape)
window[0:wSize/2] = 1.
windowedData = np.multiply(window,data);
out = gt.GammaToneApply(windowedData, forw, feed)
test1 = out[0,:]

window = np.zeros(data.shape)
window[wSize/2:wSize] = 1.
windowedData = np.multiply(window,data);
out = gt.GammaToneApply(windowedData, forw, feed)
test2 = out[0,:]

window = np.zeros(data.shape)
window[0:wSize] = 1.
windowedData = np.multiply(window,data);
out = gt.GammaToneApply(windowedData, forw, feed)
testtot = out[0,:]

plt.plot(t[0:2000],test1[0:2000])
plt.plot(t[0:2000],test2[0:2000])
plt.plot(t[0:2000],testtot[0:2000])
plt.plot(t[0:2000],test1[0:2000]+test2[0:2000])

#for i in range(nEl):
#    plt.plot(t[0:2000],out[i,:][0:2000])
#plt.plot(t[0:2000],out[0,:][0:2000])
plt.show()