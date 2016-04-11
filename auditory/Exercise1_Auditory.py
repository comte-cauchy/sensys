# -*- coding: utf-8 -*-

'''
Program for simulating the auditory experience of a patient with a cochlear implant (CI).

You can select a WAV-file as input, the program will generate and play a WAV-file with the sound simulation of a CI patient.
Two different simulation methods have been implemented one using a fourier transfomation, one using gammatone filters.

To run the program use the command 'python Exercise1_Auditory.py' while being in the directory containing the file.
First you will be asked what method (fourier/gammatones) you want to use. Valid inputs are: 'fourier' and '1' for the fourier method or 'gamma' and '2' for the gammatone method.
The default input file has been set to tiger.wav, if you have saved the file in a different location than specified in the code you will be asked to choose an input file manually.

Ver 1.0
Linus Groner, Max Hess, 07.04.2016
'''

import GammaTones as gt
import numpy as np
import matplotlib.pyplot as plt
import thLib.sounds as thSound
import os
import scipy

def calculateStimuliGamma(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
    """
    Calculates the stimulation of the electrodes of a CI for a given sound,
    by means of gamma filters.
    
    Args:
        data: array, represents the sound file.
        sampleRate: integer, sampling rate of the sound
        nElectrodes: integer, the number of electrodes
        lowFreq: number, central frequency of lowest electrode
        highFreq: number, central frequency of highest electrode
    Returns:
        gammaResponses: matrix, each row representing stimulation over time.
        cf: array, central Frequency of the electrodes
        
    """
    gammaMethod = 'moore'
    (forward,feedback,cf,ERB,B) = gt.GammaToneMake(sampleRate,
                                                   nElectrodes,
                                                   lowFreq,
                                                   highFreq,
                                                   gammaMethod)
    gammaResponses = gt.GammaToneApply(data, forward, feedback) 
    gammaResponses = np.multiply(gammaResponses,gammaResponses)
    return gammaResponses,cf

def calculateStimuliFourier(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000):
    """
    Calculates the stimulation of the electrodes of a CI for a given sound,
    by means of a fourier transform.
    
    Args:
        data: array, represents the sound.
        sampleRate: integer, sampling rate of the sound
        nElectrodes: integer, the number of electrodes
        lowFreq: number, central frequency of lowest electrode
        highFreq: number, central frequency of highest electrode
    Returns:
        res: matrix, each row representing stimulation over time.
        centralF: array, central Frequency of the electrodes
        
    """
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
        
    res = np.zeros([nElectrodes,nData])
    for i in range(len(centralF)):
        tmp = ftData[max(0,lowFidx[i]):min(highFidx[i],nData)]
        tmp = np.real(np.multiply(tmp,np.conj(tmp)))
        val = np.mean(tmp)
        res[i,:] = np.real(val)

    return res,centralF
    
def calculateStimuliWindowed(data, sampleRate, nElectrodes=21, lowFreq=200,highFreq=5000,
                             winSize=6, stepSize=0.5,method='fourier'):
    """
    Calculates the stimulation of the electrodes of a CI for a given sound,
    by means of a shifting window, where the stimulus for each window is calculated
    either by using a fourier transform (method='fourier') or gammatone filters
    (method=='gamma')
    
    Args:
        data: array, represents the sound file. 
        sampleRate: integer, sampling rate on which the sound is based
        nElectrodes: integer, the number of Electrodes in the CI
        lowFreq: number, central stimulation frequency of the lowest electrode
        highFreq: number, central stimulation frequency of the highest electrode
        winSize: number, size of the shifting window in milliseconds
        stepSize: number, how much the window is shifted in each step in milliseconds
        method: string, either 'gamma' or 'fourier', for selecting the calculation method.
    Returns:
        outData: matrix, each row representing stimulation over time.
        freq: array, central Frequency of the electrodes
    """
    nData = len(data)
    start = 0

    samplesPerWindow = int((sampleRate/1000)*winSize) #Calculate the number of sample per window of size defined above
    shift = int((sampleRate/1000)*stepSize) #Calculate the number of elements required to shift the window one 'stepSize'

    data = np.concatenate([data,np.zeros([samplesPerWindow,])]) # zero-padding end, to prevent overflow
    outData = np.zeros([nElectrodes,len(data)])

    while start < nData:
        windowData = calculateWindow(data, start, sampleRate, samplesPerWindow)
        if method == 'fourier':
            windowStimuli, freq = calculateStimuliFourier(windowData, sampleRate, nElectrodes, 
                                                          lowFreq, highFreq)
        elif method == 'gamma':
            windowStimuli, freq = calculateStimuliGamma(windowData, sampleRate, nElectrodes, 
                                                        lowFreq, highFreq)

        for i in range(nElectrodes):
            outData[i,start:start+samplesPerWindow] += windowStimuli[i,:]

        start += shift
    return outData[:,0:nData], freq

def simulateSound(stimulationData, frequencies,sampleRate):
    """
    Calculates a what a wearer of a CI might perceive for a electrode stimulation.
    
    Args:
        stimulationData: matrix, contains in each row the stimulation intensities
                         of one electrode over time
        frequencies: array, the frequencies where the electrodes are placed
        sampleRate: integer, sampling rate of the stimulationData
    Returns:
        array, representing perceived sound.
    """
    nChannels = stimulationData.shape[0]
    nData = stimulationData.shape[1]
    t = np.linspace(0,nData/sampleRate,nData)
    weightedSines = np.zeros(stimulationData.shape)
    for i in range(0,nChannels):
        f = frequencies[i]    
        sine = np.sin(2*np.pi*f*t)
        weightedSines[i] = np.multiply(sine,np.sqrt(stimulationData[i,]))
    soundData = np.sum(weightedSines,0)
    return soundData

def writeSoundToFile(outData, sampleRate, outFileName=None):
    """
    Generates a sound file from outData
    
    Args:
        otuData: array, output data representing the sound
        sampleRate: integer, sampling rate of the sound to be written
        outFileName: string, path and name of the output soundfile
    Returns:
        void
    """
    ########################################################  
    ## this section is a slightly altered copy-paste from ##
    ## thLib's setData, as it couldn't originally deal    ##
    ## with 2D arrays.                                    ##
    ########################################################    
    if not outData[0].dtype == np.dtype('int16'):
        defaultAmp = 2**13
        outData *= defaultAmp / np.max(outData)
        outData = np.int16(outData)  
     
    #########################################################  
    ## this section is a copy-paste from thLib's writeWav, ##
    ## as setData wouldn't accept Stereo sound.            ##   
    #########################################################
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
    #################################  
    ## End of copy-pasted sections ##
    #################################    
    
    #ugly hack, s.t. playing stereo is possible
    #(thLib accepts stereo files and keeps them stereo, but it is not possible
    # to construct stereo Sound objects, except from files.)
    dummySnd = thSound.Sound(outFileName);
    dummySnd.play()

def calculateWindow(data,start,sampleRate,nOut):
    """
    Takes a data array and returns an array of size 'nOut' containing the
    'nOut' samples starting at data[start], weighted with a hamming window
    
    Args:
        data: array, input data, represents a sound
        start: integer, index of the first element in data to copy
        sampleRate: integer, sampling rate of the original sound
        nOut: integer, number of elements to be copied
    Returns:
        array, the extracted window
    """
    windowedOut = np.zeros([nOut,1]) 
    windowedOut[0:nOut,0] = np.hamming(nOut)*data[start:start+nOut]
    return windowedOut.reshape(len(windowedOut))   
    
def main():
    nElectrodes = 21    #number of electrodes
    lowFreq = 200       #lower frequency boudary [Hz]
    highFreq = 5000     #higher frequency boudary [Hz]
    winSize = 6         #window size [ms]
    stepSize = .5       #step size [ms]
    inputMethod = None
    while inputMethod!='fourier' and inputMethod!='1' and inputMethod!='gamma' and inputMethod!='2':
        inputMethod = input('Please enter the method you want to use (fourier,1/gamma,2): ') #valid inputs: ('fourier','1'), ('gamma','2')
        if inputMethod=='fourier' or inputMethod=='1':
            method = 'fourier'
        elif inputMethod=='gamma' or inputMethod=='2':
            method = 'gamma'
    
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
        
    if snd.data.ndim<2:
        snd.data = np.reshape(snd.data,(len(snd.data),1))
    soundData = np.zeros(snd.data.shape)     
    for lrIdx in range(snd.data.shape[1]):
        stimuli, freq = calculateStimuliWindowed(snd.data[:,lrIdx], sampleRate, nElectrodes, lowFreq, highFreq,
                             winSize, stepSize, method)
    ##################################################################
    ## At this point, stimuli contains the stimulation intensities, ##
    ## one row per electrode. Based on this, the sound a wearer of  ##
    ## a CI might percieve is estimated.                            ##
    ##################################################################
    for lrIdx in range(snd.data.shape[1]):        
        soundData[:,lrIdx] = simulateSound(stimuli, freq, 
                                               sampleRate)
    writeSoundToFile(soundData, sampleRate,outFileName)

if __name__=='__main__':
    main()