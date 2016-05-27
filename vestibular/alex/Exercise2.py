#!/usr/bin/python
'''
'''
# Import required packages
import numpy as np
import scipy.signal as ss
import matplotlib.pylab as mp
import getXsens
 
def vestibular():
	''' 
	'''
	#define constants	
	T1 = 0.01 #sec
	T2 = 5 #sec
	samplingFreq = 50. # Hz
#	dt = 1./samplingFreq
	radius = 3.2 #millimeter
	onDirScc = np.array([[0.32269, -0.03837, -0.94573]]) # from wikibook
	
	
	#read input
	xSensData = getXsens.getXSensData('Walking_02.txt', ['Gyr'])
	
	inRot = xSensData[1]
	inRot = np.column_stack((inRot[:,0],-inRot[:,2],inRot[:,1])) # reorder axes, for consistency
	
	inAcc = xSensData[1]
	inAcc = np.column_stack((inAcc[:,0],-inAcc[:,2],inAcc[:,1])) # reorder axes, for consistency
	
	t = np.arange(0,inRot.shape[0]/samplingFreq,1./samplingFreq)

	#TODO:rotate input, s.t. inAcc[0,:] aligns w/ [0,0,-9.81]...
	gravity = inAcc[0,:]

	#Test: inAcc[0,:] should be ~[0,0,-9.81]
		
	# 1. cupular deplacement 
	#determine scalar rotation w.r.t on-direction of SCC - should pay respect to reid's line
	velRot = (onDirScc*inRot).sum(axis=1)
	
	
	# Define transfer function
	num = [T1*T2]
	den = [T1*T2,T1+T2,1]
	system = ss.lti(num, den)

	# Simulate and plot outSignal
	tout, delta, xout = ss.lsim(system, velRot, t)
	
#	#plot isplacement
#	print(delta*np.pi*radius)
#	mp.plot(tout, delta*np.pi*radius)
#	mp.show()
	
	maxDisplaceCupula = max(delta)*np.pi*radius
	minDisplaceCupula = min(delta)*np.pi*radius
	
	#2. Acceleration
	#the input should already be rotated, the data is ready to read off.
	
	maxAccOnDir = max(inAcc[:,1])
	minAccOnDir = min(inAcc[:,1])
	
	# output
	
	cupulaFile = open('CupularDisplacement.txt', 'w')
	cupulaFile.write("//cupular displacement [mm]\n//min max\n%.6f %.6f\n"%(minDisplaceCupula, maxDisplaceCupula))
	cupulaFile.close()
	
	accelFile = open('MaxAcceleration.txt', 'w')
	accelFile.write("//maximum acceleration [m/s^2]\n//min max\n%.6f %.6f\n"%(minAccOnDir, maxAccOnDir))
	accelFile.close()
	
	
if __name__ == '__main__':
	vestibular()
