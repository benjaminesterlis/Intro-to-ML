from numpy import linalg as LA
import numpy as np
import random

def K_NNeighbor(ImageSet,labelsVector,QImg):
	#if True:
	#	return random.randint(0,10) #gives 0.1 accu = 1/10

	label = 0
	IS = np.array(ImageSet)
	QI = np.array([QImg for i in range(len(ImageSet))])
	I = np.subtract(IS,QI)
	indexlist = np.argsort(LA.norm(I,axis=1))
	maxindeces = [ 0 for i in range(100)]
	for k in range(1,101):
		k_indeces = indexlist[:k]
		#for i in k_indeces:
		#	label += labelsVector[i]
		#return (int)(((label+0.0)/k)+0.5)
		#return (int)(label/k)
		accu = [0 for i in range(10)]
		for i in k_indeces:
			accu[(int)(labelsVector[i])] += 1
		maxind = 0
		for i in range(1,10):
			if(accu[i]>accu[maxind]):
				maxind = i
		maxindeces[k-1] = maxind
	return maxindeces