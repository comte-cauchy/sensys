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

