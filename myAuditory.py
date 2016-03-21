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
    return gammaResponses

def simulateSound(stimulationData):
    soundData = np.sum(stimulationData,0)
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
    dummySnd = thSound.Sound(outFileName);
    dummySnd.play()
    
    
def main():
    inFileName = './SoundData/tiger.wav'
    outFileName = './tiger_out.wav'
    
    snd = thSound.Sound(inFileName)
    
    sampleRate = snd.rate
    nElectrodes = 21
    lowFreq = 200
    highFreq = 5000
    

    if snd.data.ndim<2:
        snd.data = np.reshape(snd.data,(len(snd.data),1))
    
    soundData = np.zeros(snd.data.shape)     
    for lrIdx in range(snd.data.shape[1]):
        gammaResponse = calculateStimuli(snd.data[:,lrIdx], sampleRate, nElectrodes, lowFreq, highFreq)
        soundData[:,lrIdx] = simulateSound(gammaResponse)
    writeSoundToFile(soundData, sampleRate,outFileName)

if __name__=='__main__':
    main()