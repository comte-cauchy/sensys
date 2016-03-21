import thLib
import numpy as np
import matplotlib.pyplot as plt

def pSpect(data, rate):
    #Calculation of power spectrum and corresponding frequencies
    nData = len(data)
    fftData = np.fft.fft(data)
    PowerSpect = fftData * fftData.conj() / nData
    freq = np.arange(nData) * float(rate) / nData
    return (np.real(PowerSpect), freq)


sampleRate = 100000
freq = 1000
amp = 1

t = np.arange(0,0.01,1/sampleRate)
signal =  amp*np.sin(2*np.pi*freq*t)
#plt.plot(t,signal)
#plt.show()
signal[0:200] = 0
signal[401:] = 0
window = np.hamming(201)
signal[200:401] *= window
Pxx, freq = pSpect(signal, sampleRate)
Pxx
#Pxx[0:199] = 0
#Pxx[400:] = 0
plt.plot(signal)
plt.show()
plt.plot(freq, Pxx)
plt.show()