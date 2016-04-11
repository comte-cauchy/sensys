# -*- coding: utf-8 -*-
"""
Program for simulating a cochlea implant. The stimulation strength for each
(simulated) electrode is determined through a Fourier Transform.

The input has to be a WAV-file.
The output is optional; if selected than the simulated response is written into
a WAV-file.

If you look for an even simpler solution to the problem, check out "Quick_CI.py".
A more complex solution, which also allows for more variability in the step-
and write-size of blocks, is found in "simulate_CI_overlap.py"
"""

"""
Ver 1.3
ThH, July 2012

Changes:
    1.1 replace Qt by wxpython, and a nifty - if intricate - progressbar
    1.2 allow "main" to be called with a filename. Important for testing.
    1.3 Comments added, structure improved
"""

import sys
import os

import numpy as np
import thLib.sounds as Sound

def progressbar(it, prefix = "", size = 60):
    '''Shows a progress-bar on the commandline.
    This has the advantage that you don't need to bother with windows
    managers. Nifty coding!'''

    count = len(it)
    def _show(_i):
        # Helper function to print the desired information line.

        x = int(size*_i/count)
        sys.stdout.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), _i, count))
#        sys.stdout.flush()
    
    _show(0)
    for i, item in enumerate(it):
        yield item
        _show(i+1)
    sys.stdout.write("\n")
    sys.stdout.flush()


class CochleaImplant:
    """Definition of the cochela implant paramterers """

    def __init__(self, inSound):
                
        # Set the analysis-parameters
        winCalcSize_ms = 25   # [ms]; default 10. For the calculation of the FFT
        winStepSize_ms = 5    # default 3. 

        self.Electrodes_num = 21
        freqRange_lower = 100  # [Hz]; default 20
        freqRange_upper = 5000  # [Hz]
        
        # Initial calculations
        # For a symmetrical window, an even number of points are required -> "2", "0.5"
        self.winCalcSize_pts  = int(2*round(0.5*winCalcSize_ms * inSound.rate/1000))
        self.winStepSize_pts  = int(2*round(0.5*winStepSize_ms * inSound.rate/1000))

        # Calculate the location of the electrodes
        self.sample_freqs = np.logspace(np.log10(freqRange_lower),np.log10(freqRange_upper), self.Electrodes_num+1)
            
        self.dataNum = len(inSound.data)    
                

def pSpect(data, rate):
    '''Calculation of power spectrum and corresponding frequencies, using a Hamming window'''
    nData = len(data)
    window = np.hamming(nData)
    fftData = np.fft.fft(data[:,0]*window)
    PowerSpect = fftData * fftData.conj() / nData
    freq = np.arange(nData) * float(rate) / nData
    return (np.real(PowerSpect), freq)

def calcTransform(ci, inSound):
   ''' Step through the whole file to calculate
           i) the stimulation parameters, and
           ii) the reconstructed output
   '''

   #Allocate the required memory
   StimStrength = np.zeros(ci.dataNum)
   ReAssembled = np.zeros(ci.dataNum)
    
   time = np.arange(ci.winStepSize_pts)/float(inSound.rate)
   for ii in progressbar(range(1, ci.dataNum - ci.winCalcSize_pts + 2, ci.winStepSize_pts), 
                         'Computing ', 25):

       # Calculate PowerSpectrum
       Pxx, freq = pSpect(inSound.data[ii:(ii+ci.winCalcSize_pts-1)], inSound.rate)

       # set averaging ranges, only on the first loop
       # I have to do this here, since I need "freq" 
       if ii == 1:
           average_freqs = np.zeros([len(freq), ci.Electrodes_num])
           stim_data = np.zeros([ci.Electrodes_num,len(time)])
           
           for jj in range(ci.Electrodes_num):
                average_freqs[((freq>ci.sample_freqs[jj]) * (freq<ci.sample_freqs[jj+1])),jj] = 1
                stim_freq = np.mean(ci.sample_freqs[jj:(jj+2)])
                # "stim_data" are the sine-waves corresponding to each electrode stimulation
                stim_data[jj,:] = np.sin(2*np.pi*stim_freq*time)	       
                    
       # Average over the requested bins
       # The square root has to be taken, to get the amplitude
       StimStrength = np.sqrt(Pxx).dot(average_freqs)
       
       # Reassemble the output
       NewStim = StimStrength.dot(stim_data)
       ReAssembled[ii:(ii+ci.winStepSize_pts)] = NewStim           
           
   
   # Return the output as normalized 32-bit integer
   outFloat = (ReAssembled-np.min(ReAssembled))
   outSound = (outFloat/np.max(outFloat) * (2**31-1)).astype(np.int32)
   return Sound.Sound(inData = outSound.astype(np.float64), inRate = inSound.rate)

def main(inFile = None):
    ''' Main file. It
    * selects the sound source, and gets the data
    * defines the cochlear implant
    * calculates the simulated sound
    * writes the data to an out-file
    '''

    inSound = Sound.Sound(inFile)
    # inSound.play()
    newCi = CochleaImplant(inSound)
    ciSound = calcTransform(newCi, inSound)
    
    # Determine the name of the out-file, write and play it
    inFile = os.path.basename(str(inSound.source))
    inBase, inExt = inFile.split('.')
    fullOutFile = inBase + '_out.' + inExt
    ciSound.source = fullOutFile
    
    ciSound.writeWav(fullOutFile)    
    ciSound.play()
   

if __name__ == '__main__':
    main()