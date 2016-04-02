import numpy as np

def calculateStimuliFourier(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
    nData = data.shape[0]
    
    centralF = np.linspace(lowFreq,highFreq,nElectrodes)
    
    lowFidx = centralF - (centralF[1]-centralF[0])/2
    lowFidx = np.real(((lowFidx/(sampleRate/2))*nData/2)).astype(int)
    highFidx = centralF + (centralF[1]-centralF[0])/2
    highFidx = np.real(((highFidx/(sampleRate/2))*nData/2)).astype(int)
        
    ftData = np.fft.fft(data)  
        
    altRes = np.zeros([nElectrodes,nData])
    for i in range(len(centralF)):
        tmp = ftData[max(0,lowFidx[i]):min(highFidx[i],nData)]
        tmp = np.real(np.multiply(tmp,np.conj(tmp)))
        val = np.mean( np.sqrt( (tmp) ) )
        altRes[i,:] = np.real(val)

    return altRes,centralF

