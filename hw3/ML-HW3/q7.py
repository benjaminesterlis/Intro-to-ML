from hw3 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class Quad_Kernel_SVM():
    def __init__(self, c ,k, max_iter):
        self.k = k
        self.c = c
        self.M
        self.max_iter = max_iter

    def Quad_K(self, x, y):
        return np.square(1+ np.dot(x, y))
    def train(self, training_data, training_labels, n):
        self.M = np.zeros((self.K, self.max_iter))
        indx = np.random.RandomState(0).choice(xrange(len(train_data)), size=max_iter)
        samp  = training_data[indx]
        lab = training_labels[indx]
        f = lambda (i,j): Quad_K(samp[i], lab[j])
        Ker = [map(f,[(i, j) for i in range(j + 1)]) for j in xrange(self.max_iter)]
        for j in xrange(len(samp)):
            self.M = self.M - n * self._gradient(training_labels[j],j)


    def _gradient(self, yi, indx):
        max_arg = np.argmax([(self._dot_ker(self.M[j] - self.M[int(yi)], indx) + int(yi != j)) for j in xrange(self.K)])
        grad = np.zeros(self.K)
        for i in xrange(len(self.w)):
            if i != yi and i == max_arg:
                grad[i] += self.C * yi
            elif i == yi and yi != max_arg:
                grad[i] -= self.C * yi
        return grad

    def _dot_ker(self, W, j):
    	return np.dot(W, np.array((self.K[i][j] if i>j else self.K[j][i]) for i in xrange(self.max_iter)]))

    def _predict_indxex(self, x):
        return np.argmax(self.Quad_K(self.w, x))

    def _check(self, x, y):
        if x == y:
            return 1
        else:
            return 0

    def accuracy(self, samples, labels):
        l = len(samples)
        return float(np.sum([self._check(self._predict_index(samples[i]), labels[i]) for i in xrange(l)])) / l


def best_n(max_iter=5000):
    svm = SVM(9, 0.1, max_iter)
    l = {}
    cp = np.array(validation_labels, dtype=np.int32)
    for i in xrange(-10, 10):
        print "at best_n, iter:", i
        accuracy = 0
        error = 0
        for j in xrange(10):
            svm.train(train_data, train_labels, 10 ** (i))
            accuracy += svm.accuracy(validation_data, cp.copy())  / 10.0
            error += (1 - svm.accuracy(train_data, train_labels))   / 10.0
        l[accuracy] = i
        # plt.plot(i,accur,"go")
        # plt.plot(i, 1 - accur, "ro")
        # plt.plot(i, error, "r^")
    # plt.show()
    m = max(l)
    return 10 ** (l[m])


def best_C(n = 10, max_iter = 1000):
    l = {}
    for i in xrange(20):
        print("at best_C, iter: {}".format(i - 10))
        accur = 0
        error = 0
        svm = SVM(9, 0.1, max_iter)
        for j in xrange(10):
            svm.train(train_data, train_labels, n)
            accur += svm.accuracy(validation_data, validation_labels) / 10.0
            error += (1 - svm.accuracy(train_data, train_labels)) / 10.0
        l[accur] = i - 10
        # plt.plot(i,accur,"go")
        # plt.plot(i, 1 - accur, "ro")
        # plt.plot(i, error, "r^")
    # plt.show()
    m = max(l)
    return 10 ** (l[m]), max(l)

if __name__ == '__main__':
    # q6_a()
    # q6_b()
    q6_c() # acc = 0.9167
