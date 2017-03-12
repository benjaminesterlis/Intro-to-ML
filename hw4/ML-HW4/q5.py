from hw4 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics.classification import accuracy_score

class AdaBoost:
	
	def __init__(self, T):
		self.T = T
		self.N = 0
		self.D = []
		self.alphas = np.zeros(T)
		self.matrix_of_hypothese = []
		self.hypothesis = np.array([])

	def train(self,samples,labels):
		self.N = len(samples[0])
		self.D = np.ones(len(samples), dtype=np.float) / len(samples)
		print self.D 
		# sleep(2)
		length = len(samples)
		self.matrix_of_hypothese = self.calc_hypo_mat(samples,labels)

		for t in xrange(self.T):
			# print("iter:{}".format(t))
			h_t ,e_t, index, sgn = self.best_hypothesis(samples, labels)
			# print e_t
			self.alphas[t] = 0.5 * np.log((1 - e_t) / e_t)		
			self.hypothesis = np.append(self.hypothesis,h_t)
			z_t = 2 * np.sqrt((1 - e_t) * e_t)
			# print z_t, np.power(np.e, ((self.matrix_of_hypothese[index] - 0.5) * 2 * sgn) * self.alphas[t]), np.e, self.alphas[t]
			self.D = np.power(np.e, ((self.matrix_of_hypothese[index] - 0.5) * 2 * sgn) * self.alphas[t]) * self.D / z_t
			# print self.D
			# sleep(3)


	def best_hypothesis(self,samples,labels):
		min_error = 1
		min_sgn = 0
		min_index = -1
		for d in xrange(self.N):
			# print  "min error", min_error
			sgn = 1
			# print "indi", self.matrix_of_hypothese[d], "D", self.D
			error_d = np.inner(self.matrix_of_hypothese[d], self.D)
			# print "err_hyp:",error_d
			if error_d > 0.5 :
				 error_d = 1 - error_d
				 sgn = -1
			if error_d < min_error:
				min_error = error_d
				min_index = d
				min_sgn = sgn 
		return lambda X: self.Hypothesis(min_sgn, min_index, 0, X), min_error, min_index, min_sgn

	def calc_hypo_mat(self,samples,labels):
		indicate_mat = zeros((self.N, len(samples)), np.float)
		for i in xrange (len(samples[0])): 
			for j,samp in enumerate(samples):
				indicate_mat[i][j] = (samp[i] <= 0) != (labels[j] == 1)
		# print indicate_mat
		return indicate_mat


	def Hypothesis(self, sgn,index,theta,X):
		return ((np.take(X,index,axis = 1) <= theta) -0.5) * 2 * sgn


	def accuracy(self, X, labels):
		accur = zeros(self.T)
		pred = zeros(len(X))
		for t in xrange(self.T):
			pred += self.hypothesis[t](X) * self.alphas[t]
			accur[t] = accuracy_score(labels,np.sign(pred))

		return accur

	def avg_expo(self, X, labels):
		accur = zeros(self.T)
		pred = zeros(len(X))
		for t in xrange(self.T):
			pred += self.hypothesis[t](X) * self.alphas[t]
			accur[t] = np.sum(np.power(np.e,-labels *pred)) / len(X)
		return accur

#theta is 0 due to symmetry
def q5_a(T = 50):

	ada = AdaBoost(T)
	ada.train(train_data, train_labels)
	train_accur = ada.accuracy(train_data, train_labels)
	test_acuur = ada.accuracy(test_data, test_labels)
	plt.plot(range(T), train_accur, '-o', label="train accuracy")
	plt.plot(range(T), test_acuur, '-o', label="test accuracy")
	plt.legend()
	plt.axis('auto')
	plt.show()

def q5_b(T = 50):

	ada = AdaBoost(T)
	ada.train(train_data, train_labels)
	train_accur = ada.avg_expo(train_data, train_labels)
	test_acuur = ada.avg_expo(test_data, test_labels)
	plt.plot(range(T), train_accur, '-o', label="train loss")
	plt.plot(range(T), test_acuur, '-o', label="test loss")
	plt.legend()
	plt.axis('auto')
	plt.show()


if __name__ == '__main__':
	# q5_a()
	q5_b()