'''
Based on measurements of 
* linear acceleration
* angular velocity
by a xSens sensor, this program calculates
* the minimum and maximum cupular displacement in the right horizontal semi-circular canal
* the minimum and maximum linear acceleration in direction of the sensor's on-direction (to the left of the head)
* the final orientation of the wearers head in space.

Output of the cupular displacement is to the file CupularDisplacement.txt
Output of the linear acceleration is to the file MaxAcceleration.txt
All three results are printed to standard output.

By default, sensor data is read from 'Walking_02.txt' in the working directory.
Different paths can be passed to main(dataPath).

Authors: Max Hess, Linus Groner
Date: 13 May 2016
Version: 1.0
'''
from thLib import quat as thq
from thLib import vector as thv
from thLib import rotmat as thr
import getXsens as xsens
import numpy as np
import scipy.signal as ss

def sens2head(gravity):
    #using gravity as reference, this quaternion is calculated, such that 
    #it represents a rotation of data in xSens-coordinates to head-oriented coordinates.
    quat = thv.qrotate(gravity, np.array([0, 1, 0]))
    rM_quat = thq.quat2rotmat(quat)
    
    #since xSens uses an odd convention for their coordinates, we  have to rotate
    #the data by 90° around x axis
    rM_large = np.matrix([[1,0,0],
                          [0,0,-1],
                          [0,1,0]])
    
    #the aligning step and the convention-fixing step are combined and returned.
    rM_total = rM_large.dot(rM_quat)
    return thq.rotmat2quat(rM_total)

def headOrientation(angVelHead, sampleFreq):
    
    headDir = np.array([1,0,0])
    totalQuat = thq.vel2quat(angVelHead*180/np.pi,np.array([0,0,0]),sampleFreq,'bf')[-1,:]
    return thq.quat2rotmat(totalQuat).dot(headDir)

def rhSCCDisplacement(angVel, time):
    radius = 3.2e-03    
    #define transfer function
    T1 = 0.01 #value from wikibook
    T2 = 5 #value from wikibook

    #aligne with reid's line
    rM_reid = thr.R2(-15)
    
    #define orientation and size of right horizontal semicircular canal, and turn it into space-fixed coordinates
    rhScc = np.array([0.32269,-0.03837,-0.94573])
    rhSccSpace = rM_reid.dot(rhScc)
    
    #sensed angular velocity in right horizontal semicircular canal
    vel = angVel.dot(rhSccSpace)    

    numerator = [T1*T2, 0]
    denominator = [T1*T2, T1+T2, 1]
    
    scc = ss.lti(numerator,denominator)
    
    
    #calculate system response of input using transfer function defined above
    timeOut,sysResponse,timeEvol = ss.lsim(scc,vel,time)
    
    #calculate the displacement of the cupula from the system response
    displacementCupula = sysResponse*radius
    return displacementCupula
    
def main(dataPath):
    #reading in the data
    xSensData = xsens.getXSensData(dataPath, ['Counter','Acc', 'Gyr'])
    
    sampleFreq = xSensData[0]
    time =(xSensData[1].flatten()-xSensData[1][0])/sampleFreq
    sensedLinAcc = xSensData[2]
    sensedAngVel = xSensData[3]
    
    sensedGravity = sensedLinAcc[0,:]
    
    #data is transformed to account for a different coordinate frame used by xSens and for
    #the sensor being fixed to the head slightly tilted 
    toHeadQuat = sens2head(sensedGravity)        
    angVelHead = thq.quat2rotmat(toHeadQuat).dot(sensedAngVel.T).T
    linAccHead = thq.quat2rotmat(toHeadQuat).dot(sensedLinAcc.T).T
    
    #calculate the displacement of the right horizontal SCC based on the measured angular velocity
    displacement = rhSCCDisplacement(angVelHead, time)
    #calculate the linear acceleration in on-direction of the sensor
    acceleration = linAccHead.dot([0,1,0])
    
    #calculate the final head orientation in space, based on the measured angular velocities
    finalHeadDir = headOrientation(angVelHead,sampleFreq)
    
    
    #output to std out and files
    outFileName = 'CupularDisplacement.txt'
    outFile = open(outFileName, 'w')
    outString = 'Maximum Displacement: %f mm\nMinimum Displacement: %f mm\n'%(1000*max(displacement),1000*min(displacement))
    outFile.write(outString)        
    outFile.close()
    print(outString)
    
    outFileName = 'MaxAcceleration.txt'
    outFile = open(outFileName, 'w')
    outString  = 'Maximum Acceleration: %f m/s/s\nMinimum Acceleration: %f m/s/s\n'%(max(acceleration),min(acceleration))
    outFile.write(outString)        
    outFile.close()     
    print(outString)
    
    outString = 'Final Head Orientation in Space-Fixed Coordinates: (%f,%f,%f)^T\n'% (finalHeadDir[0],finalHeadDir[1],finalHeadDir[2])
    print(outString)
    
    
if __name__ == '__main__':
    main('Walking_02.txt')
     
    
    
