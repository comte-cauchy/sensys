import numpy as np
import os
from thLib import rotmat, vector, quat
from getXsens import getXSensData
import scipy.signal as ss

# choose input file & assign raw data to objects
#inFile = r'C:\Users\Max Hess\Documents\Python\Vestibular System\Walking_02.txt'
inFile = r'../MovementData/Walking_02.txt'
data = getXSensData(inFile,['Counter','Acc','Gyr'])

sampleRate = data[0]
time = (data[1].flatten()-data[1][0])/sampleRate
angVelRaw = data[3]
gravity = data[2][0]


#rotation matrix for aligning sensor coordinate system with space fixed coordinates
quaternion = vector.qrotate(gravity, np.array([0,1,0]))
rM_quat = quat.quat2rotmat(quaternion)

rM_large = rotmat.R1(90)

rM_total = rM_large.dot(rM_quat)

#rotate angular velocity data

angVel = angVelRaw.dot(rM_total.T)


#aligne with reid's line
rM_reid = rotmat.R2(-15)

#define orientation and size of right horizontal semicircular canal, and turn it into space-fixed coordinates
rhScc = np.array([0.32269,-0.03837,-0.94573])
rhSccSpace = rM_reid.dot(rhScc)
radius = 3.2e-03

#sensed angular velocity in right horizontal semicircular canal
stimulation = angVel.dot(rhSccSpace)


#define transfer function
T1 = 0.01 #value from wikibook
T2 = 5 #value from wikibook

numerator = [T1*T2, 0]
denominator = [T1*T2, T1+T2, 1]

scc = ss.lti(numerator,denominator)


#calculate system response of input using transfer function defined above
timeOut,sysResponse,timeEvol = ss.lsim(scc,stimulation,time)

#calculate the displacement of the cupula from the system response
displacementCupula = sysResponse*radius

maxDisplacement = max(displacementCupula)*1000
minDisplacement = min(displacementCupula)*1000


#write displacement to file
file = open('CupularDisplacement.txt','w')
file.write('Cupular displacement of the right horizontal semicircular canal\n\n')
file.write('Maximum Displacement: %f mm' %maxDisplacement)
file.write('\n')
file.write('Minimum Displacement: %f mm' %minDisplacement)
file.close

#display file content
file = open('CupularDisplacement.txt','r')
print(file.read())