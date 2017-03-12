from numpy import linalg as LA
import numpy as np
import random

def K_NNeighbor(ImageSet,labelsVector,QImg):
	#if True:
	#	return random.randint(0,10) #gives 0.1 accu = 1/10

	IS = np.array(ImageSet)
	QI = np.array([QImg for i in range(len(ImageSet))])
	I = np.subtract(IS,QI)
	indexlist = np.argsort(LA.norm(I,axis=1))
	labels = [ 0 for i in range(100)]
	for k in range(1,101):
		label = 0
		k_indeces = indexlist[:k]
		for i in k_indeces:
			label += labelsVector[i]
		labels[k-1] = (int)(((label+0.0)/k)+0.5)
	return labels