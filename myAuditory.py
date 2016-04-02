import GammaTones as gt
import numpy as np
import matplotlib.pyplot as plt
import thLib.sounds as thSound
import os
import scipy

def calculateStimuli(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
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
    f = np.linspace(0,sampleRate,nData)
    
    ftData = np.fft.fft(data)
    ftData = ftData[0:length(ftData)/2]
    
    iftData = np.fft.ifft(ftData)
    
    return stimuli,cf


def simulateSound(stimulationData):
    soundData = np.sum(stimulationData,0)
    return soundData

def simulateSoundSumOfSines(stimulationData, frequencies,sampleRate):
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
    windowedOut = np.zeros([nOut,1]) #allocate required memory
    windowedOut[0:samplesPerWindow,0] = data[start:start+samplesPerWindow]
    return windowedOut.reshape(len(windowedOut))   
    
def calculateStimuliWindowed(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000,
                             winSize=6, stepSize=0.5, winTailSize = 60):
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
        windowStimuli, freq = calculateStimuli(windowData, sampleRate, nElectrodes, 
                                         lowFreq, highFreq)
        for i in range(nElectrodes):
            outData[i,start:start+nOut] += windowStimuli[i,:]

        start += shift
    return outData[:,0:nData], freq

def main():
    inFileName = './SoundData/tiger.wav'
    #inFileName = './SoundData/vowels.wav'
    #inFileName = './SoundData/Mond_short.mp3'
    #inFileName=None
    outFileName = './tiger_out.wav'
    method = 'sum' #alt: 'sum', 'sumOfSines'
    stimuliMethod = 'gamma' #alt: 'fourier', 'gamma'
    snd = thSound.Sound(inFileName)
    
    sampleRate = snd.rate
    nElectrodes = 21
    lowFreq = 100
    highFreq = 5000
    winSize = 25 #ms
    stepSize = .5 #ms
    winTailSize = 6 #ms 
    

    if snd.data.ndim<2:
        snd.data = np.reshape(snd.data,(len(snd.data),1))
    soundData = np.zeros(snd.data.shape)     
    for lrIdx in range(snd.data.shape[1]):
        if stimuliMethod == 'fourier':
            stimuli, freq = calculateStimuliFourier(snd.data[:,lrIdx], sampleRate, nElectrodes, lowFreq, highFreq)
        elif stimuliMethod == 'gamma':
            #stimuli, freq = calculateStimuli(snd.data[:,lrIdx], sampleRate, nElectrodes, lowFreq, highFreq)
            stimuli, freq = calculateStimuliWindowed(snd.data[:,lrIdx], sampleRate, nElectrodes, lowFreq, highFreq,
                             winSize, stepSize, winTailSize)
        if method == 'sum':
            soundData[:,lrIdx] = simulateSound(stimuli)
        elif method == 'sumOfSines':
            soundData[:,lrIdx] = simulateSoundSumOfSines(stimuli, freq, 
                                               sampleRate)
    writeSoundToFile(soundData, sampleRate,outFileName)

if __name__=='__main__':
    main()