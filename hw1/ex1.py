from numpy import linalg as LA
import numpy as np
# import random

def K_NNeighbor(ImageSet,labelsVector,QImg ,k):
	#if True:
	#	return random.randint(0,10) #gives 0.1 accu = 1/10

	IS = np.array(ImageSet)
	QI = np.array([QImg for i in range(len(ImageSet))])
	I = np.subtract(IS,QI)
	indexlist = np.argsort(LA.norm(I,axis=1))
	k_indeces = indexlist[:k]

	accu = [0 for i in range(10)]
	for i in k_indeces:
		accu[(int)(labelsVector[i])] += 1
	maxind = 0
	for i in range(1,10):
		if(accu[i]>accu[maxind]):
			maxind = i
	return maxind