'''
Simulation of the processing of vestibular signals.
The input signals are taken from an XSENS system, which was mounted to the
side of the head and measured linear accelerations and agular velocities 
while walking in a figure-eight-loop. 
The simulation calculates
* the minimum and maximum stimulation of the right horizontal canal, assuming that Reid's plane is oriented 15 deg nose-up.
* the minimum and maximum stimulation of an otolith neuron which initially points to the left.
* the orientation of the head while walking around.

If you have "VPython" installed, the orientation of the nose is visualized in 3D.

The semicircular canals are simulated as a lowpass filter with a
time-constant of 7 sec

Compatible with Python 2 and 3.
'''

'''
Author: Thomas Haslwanter
Date:   April 2014
Ver:    1.4
'''

# Standard modules
import matplotlib.pyplot as plt
import numpy as np
import os
from thLib import rotmat, vector, ui, quat
import scipy.signal as ss
#import visual

# My own module, to read in data from the XSENS sensor
from getXsens import getXSensData

class outFile:
    '''Writing output info'''
    
    def __init__(self, outFile, outContent, outInfo):
        fh = open(outFile, 'w')
        fh.write(outContent)        
        fh.close()
        
        fh = open(outFile, 'r')
        fileContent = fh.read()
        fh.close()
        print(fileContent)
        print('{0} written to {1}\n'.format(outInfo, outFile))
        
  
class Sensor:
    '''Inertial sensor'''
    
    def __init__(self, inFile = None):

        # Get the data
        data = getXSensData(inFile, ['Counter', 'Acc', 'Gyr'])
        
        self.rate = data[0]
        self.time = (data[1].flatten()-data[1][0])/self.rate
        self.rawAcc = data[2]
        self.rawAngVel = data[3]

    def _selectInput(self):
        '''Choose the input data.'''

        dataDir = r'../MovementData'
        
        (inFile, inPath) = ui.getfile('*.txt', 'Select TXT-input: ', dataDir)
        fullInFile = os.path.join(inPath, inFile)
        print('Selection: ' + fullInFile)
        return fullInFile
    
    def toUpright(self):
        '''Rotate the sensor to align with the spatial coordinate system.'''

        # Align sensor with coordinate axes
        R_coarse = rotmat.R1(90)

        # Express the remaining rotation as a quaternion
        q_R0 = vector.qrotate(self.rawAcc[0], np.r_[0,1.,0])
        R_fine = quat.quat2rotmat(q_R0)
        
        R_total = R_coarse.dot(R_fine)
        self.spaceAngVel = self.rawAngVel.dot(R_total.T) 

        # Rotate linear acceleration into an upright, head-fixed coordinate system
        self.accReHead = self.rawAcc.dot(R_total.T)
        
class Head:
    '''Anatomical parameters for the human head'''    

    def __init__(self):
        '''Set the anatomical parameters'''
        
        # Orientation horizontal semicircular canal, right side
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
        ReidangVel = rotmat.R2(15).dot(sensor.spaceAngVel.T)

        # sensed angular velocity
        omega = ReidangVel.T.dot(self.n_HorScc)
        
        # Canal dynamics
        tout, cupulaTheta, xout = ss.lsim(self.scc, omega, sensor.time)

        # Displacement maxima
        cupula = cupulaTheta * self.r_scc
        
        sensedMaxima = {}
        sensedMaxima['deflection_Max'] = max(cupula)
        sensedMaxima['deflection_Min'] = min(cupula)
        
        # Then for linear acceleration ----------------------------
        onDir = np.array([0., 1., 0.])
        accSensed = sensor.accReHead.dot(onDir)
        
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
    q_Head = quat.vel2quat(xSens.spaceAngVel*180/np.pi, [0., 0., 0.], xSens.rate, 'bf')
    plt.plot(xSens.time, q_Head[:,1:4])
    plt.xlabel('Time [sec]')
    plt.ylabel('Head Orientation [quat]')
    plt.grid()
    # plt.show()
    plt.show(block=False)
    plt.pause(2)
    plt.close()
    q_final = q_Head[-1,:]
    finalNose = vector.rotate_vector([1., 0., 0.], q_final)
    
    outContent = 'Final nose orientation [x/y/z]: [%4.3f/%4.3f/%4.3f]' % (finalNose[0], finalNose[1], finalNose[2])
    outInfo = 'Nose orientation'
    outFile('Nose.txt', outContent, outInfo)    

    #try:
        #''' Draw a coordinate system
        #Note that VPython has the following conventions:
            #x ... forward
            #y ... up
            #z ... right
        #In order to have the coordinate system I am used to, I have to set the
        #variables in the following way:
            #x ... [2]
            #y ... [0]
            #z ... [1]
        #This is mighty confusing, even for me (ThH)
        #'''
        
        #cs = visual.curve(pos=[(-2, 0, 0), (2, 0, 0), 
                           #(0, 0, 0), (0, 2, 0), (0, -2, 0),
                            #(0,0,0), (0, 0, 2), (0,0,-2)])
        #horPlane = visual.curve(pos=[(-2,0,-2), (-2,0,2),(2,0,2),(2,0,-2), (-2,0,-2)])
        
        
        #xLabel = visual.label(pos=(0,0,2), text='X')
        #yLabel = visual.label(pos=(2,0,0), text='Y')
        #zLabel = visual.label(pos=(0,2,0), text='Z')
    
        ## Draw and animate the nose direction
        #ii = 0
        #q_arrow = visual.arrow(pos=(0,0,0), 
                        #axis=vector.rotate_vector([1., 0., 0.], q_Head[ii,:]),
                        #shaftwidth=2)
        #visual.scene.forward = (-0.7, -0.3, -0.7)
        #while ii<len(xSens.time)-1:
            #ii = ii+1
            #visual.rate(100)
            #dir_vector = vector.rotate_vector([1., 0., 0.], q_Head[ii,:])
            #q_arrow.axis= (dir_vector[1], dir_vector[2], dir_vector[0])
    
    #except ImportError:    
        #print('Sorry, no visulalization: VPython can not be imported.')
    
    print('Done')
        
if __name__ == '__main__':
    # Select the in-file
    inFile = r'../MovementData/Walking_02.txt'
    main(inFile)
