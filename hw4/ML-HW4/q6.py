from __future__ import division
from hw4 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def PCA(samples, k): #PCA works only if the data is normalized
	samples -= np.mean(samples, axis=0)
	U, E, Vt = np.linalg.svd(samples, full_matrices=False)
	Uk = U[:,:k]
	Ek = E[:k]
	Vk = Vt.T[:,:k]

	return Uk, Ek, Vk

def q6_a(d=100):
	samples = [train_data_unscaled[i] for i in xrange(train_data_size) if train_labels[i] == 1]
	Uk, Ek, Vk = PCA(samples, d)
	mean = np.mean(samples, axis=0)
	plt.title("mean sample")
	plt.imshow(reshape(mean, (28, 28)), interpolation='nearest')
	plt.show()
	for i in xrange(6):
		plt.subplot(3,2,i)
		if i < 5:
			plt.title("the {i}-th eigenvector".format(i=i))
		else:
			plt.title("Bonus: eigenvector (for symmetry)")	
		plt.imshow(reshape(Vk[:,i], (28, 28)), interpolation='nearest')

	plt.show()
	plt.title('Singular Values')
	plt.plot(Ek,'-o')
	plt.xlabel('Singular Value')
	plt.ylabel('Dimension')
	plt.show()

def q6_b(d=100):
	samples = [train_data_unscaled[i] for i in xrange(train_data_size) if train_labels[i] == -1]
	Uk, Ek, Vk = PCA(samples, d)
	mean = np.mean(samples, axis=0)
	plt.title("mean sample")
	plt.imshow(reshape(mean, (28, 28)), interpolation='nearest')
	plt.show()
	for i in xrange(6):
		plt.subplot(3,2,i)
		if i < 5:
			plt.title("the {i}-th eigenvector".format(i=i))
		else:
			plt.title("Bonus: eigenvector (for symmetry)")	
		plt.imshow(reshape(Vk[:,i], (28, 28)), interpolation='nearest')

	plt.show()
	plt.title('Singular Values')
	plt.plot(Ek,'-o')
	plt.xlabel('Singular Value')
	plt.ylabel('Dimension')
	plt.show()

def q6_c(d=100):
	samples = train_data_unscaled
	Uk, Ek, Vk = PCA(samples, d)
	mean = np.mean(samples, axis=0)
	plt.title("mean sample")
	plt.imshow(reshape(mean, (28, 28)), interpolation='nearest')
	plt.show()
	for i in xrange(6):
		plt.subplot(3,2,i)
		if i < 5:
			plt.title("the {i}-th eigenvector".format(i=i))
		else:
			plt.title("Bonus: eigenvector (for symmetry)")	
		plt.imshow(reshape(Vk[:,i], (28, 28)), interpolation='nearest')

	plt.show()
	plt.title('Singular Values')
	plt.plot(Ek,'-o')
	plt.xlabel('Singular Value')
	plt.ylabel('Dimension')
	plt.show()

def q6_d(d=2):
	Uk, Ek, Vk = PCA(train_data_unscaled, d)
	y = np.dot(train_data_unscaled,Vk)
	for i in xrange(len(y)):
		if train_labels[i] == 1:
			plt.plot(y[i:,0],y[i:,1],'o',color="pink")
		else:
			plt.plot(y[i:,0],y[i:,1],'o',color="cyan")
	plt.show()

def q6_e():
	i = 0
	indexes = []

	for x in xrange(len(train_labels)):
		if train_labels[x] == 1:
			i += 1
			indexes.append(x)
		if i == 2:
			break
 	i=0
	for x in xrange(len(train_labels)):
		if train_labels[x] == -1:
			i += 1
			indexes.append(x)
		if i == 2:
			break

	for d in xrange(10,51,20):
		Uk, Ek, Vk = PCA(train_data_unscaled, d)
		reconst = Uk.dot(diag(Ek).dot(Vk.T))

		for i in xrange(0,7,2):
			plt.subplot(4,2,i+1)
			plt.title('originals i={i} for d={d}'.format(d=d,i=i//2+1))
			plt.imshow(reshape(train_data_unscaled[indexes[i//2]], (28, 28)), interpolation='nearest')
			plt.subplot(4,2,i+2)
			plt.title('reconstructions i={i} for d={d}'.format(d=d,i=i//2+1))
			plt.imshow(reshape(reconst[indexes[i//2]], (28, 28)), interpolation='nearest')
		plt.show()

if __name__ == '__main__':
	# q6_a()
	# q6_b()
	# q6_c()
	# q6_d()
	q6_e()