import GammaTones as gt
import numpy as np
import matplotlib.pyplot as plt
import thLib.sounds as thSound
import os
import scipy


import numpy as np
import GammaTones as gt
def calculateStimuliGamma(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
    gammaMethod = 'moore'
    (forward,feedback,cf,ERB,B) = gt.GammaToneMake(sampleRate,
                                                   nElectrodes,
                                                   lowFreq,
                                                   highFreq,
                                                   gammaMethod)
    gammaResponses = gt.GammaToneApply(data, forward, feedback) 
    return gammaResponses,cf

def calculateStimuliFourier(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
    nData = data.shape[0]
    

    #linearly distributed frequencies
    #better for targeted window sizes and sampling frequency
    centralF = np.linspace(lowFreq,highFreq,nElectrodes)
    
    lowFidx = centralF - (centralF[1]-centralF[0])/2
    lowFidx = np.real(((lowFidx/(sampleRate/2))*nData/2)).astype(int)
    
    highFidx = centralF + (centralF[1]-centralF[0])/2
    highFidx = np.real(((highFidx/(sampleRate/2))*nData/2)).astype(int)
            
    ##alternative: logrithmically distributed frequencies
    ##more realistic, but issues with targeted window sizes and sampling frequency
    ##if this option is uncommented, the window size must be about 30ms for files
    ##with a sampling rate of 44100, otherwise the frequency channels are too close
    ##together. (multiple centered all on the same frequency when discretized)
    #expFreq = np.linspace(np.log10(lowFreq),np.log10(highFreq),nElectrodes)
    #centralF = 10**expFreq
    
    #lowFexp = expFreq - (expFreq[1]-expFreq[0])/2
    #lowFidx = 10**lowFexp
    #lowFidx = np.real(((lowFidx/(sampleRate/2))*nData/2)).astype(int)
    
    #highFexp = expFreq + (expFreq[1]-expFreq[0])/2
    #highFidx = 10**highFexp
    #highFidx = np.real(((highFidx/(sampleRate/2))*nData/2)).astype(int)
    
    ftData = np.fft.fft(data)  
        
    altRes = np.zeros([nElectrodes,nData])
    for i in range(len(centralF)):
        tmp = ftData[max(0,lowFidx[i]):min(highFidx[i],nData)]
        tmp = np.real(np.multiply(tmp,np.conj(tmp)))
        val = np.mean( np.sqrt( (tmp) ) )
        altRes[i,:] = np.real(val)

    return altRes,centralF
    
def calculateStimuliWindowed(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000,
                             winSize=6, stepSize=0.5, winTailSize = 60,method='fourier'):
    nData = len(data)
    start = 0

    nOut = int((sampleRate/1000)*winTailSize) #Calculate the number of sample per window of size defined above
    samplesPerWindow = int((sampleRate/1000)*winSize) #Calculate the number of sample per window of size defined above
    shift = int((sampleRate/1000)*stepSize) #Calculate the number of elements required to shift the window one 'stepSize'

    data = np.concatenate([data,np.zeros([nOut,])]) # zero-padding end, to prevent overflow
    outData = np.zeros([nElectrodes,len(data)])

    while start < nData:
        windowData = calculateWindow(data, start, sampleRate, samplesPerWindow, 
                                     nOut)
        if method == 'fourier':
            windowStimuli, freq = calculateStimuliFourier(windowData, sampleRate, nElectrodes, 
                                                          lowFreq, highFreq)#note the af.
        elif method == 'gamma':
            windowStimuli, freq = calculateStimuliGamma(windowData, sampleRate, nElectrodes, 
                                                        lowFreq, highFreq)#note the ag. 

        for i in range(nElectrodes):
            outData[i,start:start+nOut] += windowStimuli[i,:]

        start += shift
    return outData[:,0:nData], freq

def simulateSound(stimulationData, frequencies,sampleRate):
    nChannels = stimulationData.shape[0]
    nData = stimulationData.shape[1]
    t = np.linspace(0,nData/sampleRate,nData)
    weightedSines = np.zeros(stimulationData.shape)
    for i in range(0,nChannels):
        f = frequencies[i]    
        sine = np.sin(2*np.pi*f*t)
        weightedSines[i] = np.multiply(sine,stimulationData[i,])
    soundData = np.sum(weightedSines,0)
    return soundData
        


def writeSoundToFile(outData, sampleRate, outFileName=None):
    
    if not outData[0].dtype == np.dtype('int16'):
        defaultAmp = 2**13
        outData *= defaultAmp / np.max(outData)
        outData = np.int16(outData)  
        
    ### this section is a copy-paste from thLib's writeWav, as it can't handle stereo sound 
    if outFileName is None:
        (outFile , outDir) = ui.savefile('Write sound to ...', '*.wav')            
        if outFile == 0:
            print('Output discarded.')
            return 0
        else:
            outFileName = os.path.join(outDir, outFile)
    else:
        outDir = os.path.abspath(os.path.dirname(outFileName))
        outFile = os.path.basename(outFileName)
    #        outFileName = tkFileDialog.asksaveasfilename()

    scipy.io.wavfile.write(str(outFileName), int(sampleRate), outData)
    print('Sounddata written to ' + outFile + ', with a sample rate of ' + str(sampleRate))
    print('OutDir: ' + outDir)
    
    ###    
    
    #ugly hack, s.t. playing stereo is possible
    #(thLib accepts stereo files and keeps them stereo, but it is not possible
    # to construct stereo Sound objects, except from files.)
    dummySnd = thSound.Sound(outFileName);
    dummySnd.play()

def calculateWindow(data,start,sampleRate,samplesPerWindow,nOut): #Takes a data array and slices it into overlapping windows of size 'winSize' [ms] with a windowshift of 'stepSize' [ms]
    #numWindows = (nData-samplesPerWindow)/shift+1
    nData = len(data)
    windowedOut = np.zeros([nOut,1]) #allocate required memory
    windowedOut[0:samplesPerWindow,0] = np.hamming(samplesPerWindow)*data[start:start+samplesPerWindow]
    return windowedOut.reshape(len(windowedOut))   
    
def main():
    nElectrodes = 21
    lowFreq = 200 #hz
    highFreq = 5000 #hz
    winSize = 6 #ms
    stepSize = .5 #ms    
    method = 'fourier' #alt: 'fourier', 'gamma'
    
    inFileName = './SoundData/tiger.wav'
    #inFileName = './SoundData/vowels.wav'
    #inFileName = './SoundData/Mond_short.wav'    

    if (('inFileName' in locals()) or ('inFileName' in globals())) and inFileName != None:
        snd = thSound.Sound(inFileName)
    else:
        snd = thSound.Sound()
        inFileName = snd.source        

    sampleRate = snd.rate    

    outStem, outSuffix = inFileName.rsplit('.',1)
    outFileName = outStem + '_out.wav'    
        
    if method == 'gamma':
        winTailSize = 5*winSize
    else:
        winTailSize = winSize
        
    if snd.data.ndim<2:
        snd.data = np.reshape(snd.data,(len(snd.data),1))
    soundData = np.zeros(snd.data.shape)     
    for lrIdx in range(snd.data.shape[1]):
        stimuli, freq = calculateStimuliWindowed(snd.data[:,lrIdx], sampleRate, nElectrodes, lowFreq, highFreq,
                             winSize, stepSize, winTailSize,method)
        
        
    for lrIdx in range(snd.data.shape[1]):        
        soundData[:,lrIdx] = simulateSound(stimuli, freq, 
                                               sampleRate)
    writeSoundToFile(soundData, sampleRate,outFileName)

if __name__=='__main__':
    main()
    
