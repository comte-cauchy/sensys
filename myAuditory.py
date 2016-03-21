import GammaTones as gt
import numpy as np
import matplotlib.pyplot as plt
import thLib.sounds as thSound

sampleRate = 16000
freq = [50, 120]
amp = [1, 0.4]
t = np.arange(0,10,1./sampleRate)
data =  amp[0]*np.sin(2*np.pi*freq[0]*t) + amp[1]*np.sin(2*np.pi*freq[1]*t) + np.random.randn(len(t))
snd = thSound.Sound('./SoundData/tiger.wav')

#ata = np.zeros((sampleRate*25e-3))     # create a 25ms input
#data[[1, 100, 200, 300]] = 1    # make a click train
data[0] = 1


(forward,feedback,cf,ERB,B) = gt.GammaToneMake(sampleRate, 5, 1, 
                                              8000, 
                                              'moore')


#for nC in range(100):
#    print(cf[nC],':',cf[nC]-ERB[nC]/2,'bis',cf[nC]+ERB[nC]/2)
    


y = gt.GammaToneApply(snd.data[:,0], forward, feedback)

chanId = 0
testy = np.sum(y,0)
snd.setData(testy, snd.rate)
snd.play()
plt.plot(cf,np.ones(len(cf)),'x')
plt.plot(cf - ERB,.8*np.ones(len(cf)),'x')
plt.plot(cf+ERB,1.2*np.ones(len(cf)),'x')
plt.plot(0,0,'x')
plt.plot(0,1,'x')

plt.show()
#plt.plot(np.arange(y.shape[1]),[chanId])
#plt.show()

