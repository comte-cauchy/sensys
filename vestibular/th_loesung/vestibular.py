'''
Simulation of the processing of vestibular signals.
The input signals are taken from an XSENS system, which was mounted to the
side of the head and measured linear accelerations and agular velocities 
while walking in a figure-eight-loop. 
The simulation calculates
* the minimum and maximum stimulation of the right horizontal canal, assuming that Reid's plane is oriented 15 deg nose-up 
* the minimum and maximum stimulation of the otolith neuron which initially points to the left
* the orientation of the head while walking around.

The orientation of the nose is visualized in 3D.

The semicircular canals are simulated as a lowpass filter with a
time-constant of 7 sec
'''

'''
Author: Thomas Haslwanter
Date:   June 2012
Ver:    1.2
'''

import matplotlib.pyplot as mpl # Grab MATLAB plotting functions
import scipy.signal as ss
import numpy as np
import os
from thLib import RotMat, quat_funcs, UI

import visual

class outFile:
    '''Writing output info'''
    
    def __init__(self, outFile, outContent, outInfo):
        fh = open(outFile, 'w')
        fh.write(outContent)        
        fh.close()
        
        fh = open(outFile, 'r')
        fileContent = fh.read()
        fh.close()
        print fileContent
        print outInfo + ' written to ' + outFile + '\n'
        
  
class Sensor:
    '''Inertial sensor'''
    
    def __init__(self, inFile = None):

        # Select the data columns
        colTime = 0
        colAcc = [1,2,3]
        colAngVel = [4,5,6]
        
        self.rate = 50

        if inFile == None:
            inFile = self._selectInput()
        if os.path.exists(inFile):    
            self.source = inFile
        else:
            print inFile + ' does NOT exist!'

        # Read data
        try:
            # fullInFile = os.path.join(dataDir, inFile)
            fh = open(inFile, 'r')
        except IOError:
            print('Could not open ' + inFile)        
            
        inData = np.genfromtxt(fh, skip_header=5)

        # Select the required columns
        self.time = (inData[:,colTime]-inData[0,colTime])/self.rate                
        self.rawAcc = inData[:,colAcc]
        self.rawAngVel = inData[:,colAngVel]

    def _selectInput(self):
        '''Choose the input data.'''

        dataDir = r'C:\Users\p20529\Documents\Teaching\ETH\CSS\Exercises\Ex_VestibularProcessing\Data'
        
        (inFile, inPath) = UI.getfile('*.txt', 'Select TXT-input: ', dataDir)
        fullInFile = os.path.join(inPath, inFile)
        print 'Selection: ' + fullInFile
        return fullInFile
    
    def toUpright(self):
        '''Rotate the sensor to align with the spatial coordinate system.'''

        x = 1
        g0 = self.rawAcc[0,:]
        g0_ideal = np.array([0,1,0])*9.81
        
        g0_n = g0/np.sqrt(np.sum(g0**2))
        g0i_n = g0_ideal/np.sqrt(np.sum(g0_ideal**2))
        
        # axis of rotation
        n_R0 = np.cross(g0_n, g0i_n)
        n_R0 = n_R0/np.sqrt(np.sum(n_R0**2))
        
        # magnitude of rotation
        alpha_R0 = np.math.acos(np.dot(g0_n, g0i_n))
        
        # express the rotation as a quaternion
        q_R0 = n_R0 * np.sin(alpha_R0/2.)
        R_sensorToUpright = RotMat.Quat2Rotmat(q_R0)
        
        R_total = np.dot(RotMat.R1(90), R_sensorToUpright)
        self.spaceAngVel = np.dot(self.rawAngVel, R_total.transpose()) 

        # Rotate linear acceleration into an upright, head-fixed coordinate system
        self.accReHead = np.dot(self.rawAcc, R_total.transpose())
        

class Head:
    '''Anatomical parameters for the human head'''    

    def __init__(self):
    # orientation horizontal semicircular canal, right side
        self.n_HorScc = np.array([.365,	 .158,	-.905])        
        self.r_scc = 3.2e-3
        
        # Canal dynamics
        T1 = 0.01
        T2 = 5
        num = [T1*T2, 0]
        den = [T1*T2, T1+T2, 1]
        
        self.scc = ss.lti(num,den)

    def vestibularStimulation(self, sensor):
        '''Stimulation of canals and otoliths.'''

        # First for the angular velocity -----------------------
        ReidangVel = np.dot(RotMat.R2(15), sensor.spaceAngVel.transpose())

        # sensed angular velocity
        omega = np.dot(ReidangVel.transpose(), self.n_HorScc)
#        mpl.plot(sensor.time, omega)
#        mpl.show()
#        raw_input()    
        
        # Canal dynamics
        tout, cupulaTheta, xout = ss.lsim(self.scc, omega, sensor.time)
        #mpl.hold
        #mpl.plot(tout, yout,'r')

        # Displacement maxima
        cupula = cupulaTheta * self.r_scc
        
        sensedMaxima = {}
        sensedMaxima['deflection_Max'] = max(cupula)
        sensedMaxima['deflection_Min'] = min(cupula)
        
        # Then for linear acceleration ----------------------------
        onDir = np.array([0., 1., 0.])
        accSensed = np.dot(sensor.accReHead, onDir)
        
        # Acceleration maxima
        sensedMaxima['acc_Max'] = max(accSensed)
        sensedMaxima['acc_Min'] = min(accSensed)
        
        return sensedMaxima
        

def main(inFile = None):
    '''Main simulation file.'''
        
    # define the sensor, and get the data
    xSens = Sensor(inFile)
    xSens.toUpright()
    
    # Define the head
    head = Head()
    
    # Calculate the stimulation of the canals and otoliths
    sensedMaxima = head.vestibularStimulation(xSens)
    
    # Write the cupula data to an out-file
    outContent = 'Max Displacement [m]: %8.6f \n' % sensedMaxima['deflection_Max']
    outContent += 'Min Displacement [m]: %8.6f' % sensedMaxima['deflection_Min']
    outInfo = 'Cupular maxima'
    outFile('CupularDisplacement.txt', outContent, outInfo)
    
    # Write the acceleration data to an out-file
    outContent = 'Max Acceleration [m/s^2]: %8.6f \n' % sensedMaxima['acc_Max']
    outContent += 'Min Acceleration [m/s^2]: %8.6f' % sensedMaxima['acc_Min']
    outInfo = 'Acceleration maxima'
    outFile('MaxAcceleration.txt', outContent, outInfo)
    
    # Orientation of the head
    q_Head = quat_funcs.vel2quat(xSens.spaceAngVel*180/np.pi, [0., 0., 0.], xSens.rate, 'bf')
    mpl.plot(xSens.time, q_Head[:,1:4])
    mpl.grid()
    # mpl.show()
    mpl.show(block=False)
    mpl.pause(1)
    q_final = q_Head[-1,:]    
    finalNose = quat_funcs.rotate_vector([1., 0., 0.], q_final)
    
    outContent = 'Final nose orientation [x/y/z]: [%4.3f/%4.3f/%4.3f]' % (finalNose[0], finalNose[1], finalNose[2])
    outInfo = 'Nose orientation'
    outFile('Nose.txt', outContent, outInfo)    

    try:
        ''' Draw a coordinate system
        Note that VPython has the following conventions:
            x ... forward
            y ... up
            z ... right
        In order to have the coordinate system I am used to, I have to set the
        variables in the following way:
            x ... [2]
            y ... [0]
            z ... [1]
        This is mighty confusing, even for me (ThH)
        '''
        
        cs = visual.curve(pos=[(-2, 0, 0), (2, 0, 0), 
                           (0, 0, 0), (0, 2, 0), (0, -2, 0),
                            (0,0,0), (0, 0, 2), (0,0,-2)])
        horPlane = visual.curve(pos=[(-2,0,-2), (-2,0,2),(2,0,2),(2,0,-2), (-2,0,-2)])
        
        
        xLabel = visual.label(pos=(0,0,2), text='X')
        yLabel = visual.label(pos=(2,0,0), text='Y')
        zLabel = visual.label(pos=(0,2,0), text='Z')
    
        # Draw and animate the nose direction
        ii = 0
        q_arrow = visual.arrow(pos=(0,0,0), 
                        axis=quat_funcs.rotate_vector([1., 0., 0.], q_Head[ii,:]),
                        shaftwidth=2)
        visual.scene.forward = (-0.7, -0.3, -0.7)
        while ii<len(xSens.time)-1:
            ii = ii+1
            visual.rate(100)
            dir_vector = quat_funcs.rotate_vector([1., 0., 0.], q_Head[ii,:])
            q_arrow.axis= (dir_vector[1], dir_vector[2], dir_vector[0])
    
    except ImportError:    
        print 'Sorry, no visulalization: VPython can not be imported.'
    
    print 'done'    
        

if __name__ == '__main__':
    # Select the in-file
    inFile = r'C:\Users\p20529\Documents\Teaching\ETH\CSS\Exercises\Ex_VestibularProcessing\Data\Walking_02.txt'
    main(inFile)