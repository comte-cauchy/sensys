import numpy as np
import thLib.sounds as thSound
import GammaTones as gt

inSound = thSound.Sound(inFile='SoundData/tiger.wav')

data = inSound.data[:,0]
sampleRate = inSound.rate
winSize = 6
stepSize = 0.5

def calculateStimuli(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
    gammaMethod = 'moore'
    (forward,feedback,cf,ERB,B) = gt.GammaToneMake(sampleRate,
                                                   nElectrodes,
                                                   lowFreq,
                                                   highFreq,
                                                   gammaMethod)
    gammaResponses = gt.GammaToneApply(data, forward, feedback) 
    return gammaResponses

def calculateWindow(data,start,sampleRate,samplesPerWindow,nOut): #Takes a data array and slices it into overlapping windows of size 'winSize' [ms] with a windowshift of 'stepSize' [ms]
    #numWindows = (nData-samplesPerWindow)/shift+1
    windowedOut = np.zeros([1,nOut]) #allocate required memory
    windowedOut[0:samplesPerWindow] = data[start:samplesPerWindow]
    #counter = 0 #define a counter
    #start = 0 #define starting element of window
    #end = samplesPerWindow #define last element of window
    
    #while end < nData:
        #window = data[start:end]
        #calculatedWindow = calculateStimuli(window,sampleRate)
        #start = start + shift
        #end = end + shift
        #counter = counter + 1
        ##print(counter)
    
    return windowedOut

def calculateStimuliWindowed(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000,
                             winSize=6,stepSize=0.5,nOut):
    nData = len(data)
    outData = zeros(data.shape)
    start = 0

    samplesPerWindow = int((sampleRate/1000)*winSize) #Calculate the number of sample per window of size defined above
    shift = int((sampleRate/1000)*stepSize) #Calculate the number of elements required to shift the window one 'stepSize'

    data = np.concatenate([data,zeros([1,samplesPerWindow])]) # zer-padding end, to prevent overflow
    
    while start < nData:
        windowData = calculateWindow(data, start, sampleRate, samplesPerWindow, 
                                    nOut)
        windowStimuli = calculateStimuli(windowData, sampleRate, nElectrodes, 
                                        lowFreq, highFreq)
        for i in range(nElectrodes):
            outData[i,start:start+samplesPerWindow] += windowStimuli
                  
        start += shift



#out = calculateWindows(data,sampleRate)
#print(out.shape)
#calcOut = np.zeros([out.shape[0],out.shape[1],21])
#print(calcOut.shape)
#print(calcOut[0])
#print(calcOut[0,0])
#print(calcOut[0,0,0])

#counter = 0
#nData = len(out)

#while counter < nData:
    #calcOut[counter] = calculateStimuli(out[counter],sampleRate)
    #counter = counter + 1
    #print (counter)