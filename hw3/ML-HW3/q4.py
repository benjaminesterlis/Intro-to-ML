from hw3 import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections


class SVM:
    def __init__(self, K, C):
        self.C = C
        self.K = K + 1

    def train(self, training_data, training_labels, n, max_iter):
        self.w = np.zeros((self.K, len(training_data[0])))
        l = len(training_data)
        for t in xrange(max_iter):
            j = np.random.randint(l)
            assert isinstance(j, int)
            assert (0 <= j < l)
            self.w = self.w - n * self._gradient(training_data[j], training_labels[j])

    def _gradient(self, xi, yi):
        max_arg = np.argmax([(np.dot(self.w[j] - self.w[int(yi)], xi) + int(yi != j)) for j in xrange(self.K)])
        grad = np.array([self.w[i] for i in xrange(len(self.w))])
        for i in xrange(len(self.w)):
            if i != yi and i == max_arg:
                grad[i] += self.C * xi
            elif i == yi and yi != max_arg:
                grad[i] -= self.C * xi
        return grad

    def _predict_index(self, x):
        return np.argmax(np.dot(self.w, x))

    def _check(self, x, y):
        if x == y:
            return 1
        else:
            return 0

    def accuracy(self, samples, labels):
        l = len(samples)
        return float(np.sum([self._check(self._predict_index(samples[i]), labels[i]) for i in xrange(l)])) / l


def best_n(max_iter=5000):
    svm = SVM(9, 0.1)
    l = {}
    cp = np.array(validation_labels, dtype=np.int32)
    for i in xrange(-10, 10):
        print "at best_n, iter:", i
        accuracy = 0
        error = 0
        for j in xrange(10):
            svm.train(train_data, train_labels, 10 ** (i), max_iter)
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
        svm = SVM(9, 0.1)
        for j in xrange(10):
            svm.train(train_data, train_labels, n, max_iter)
            accur += svm.accuracy(validation_data, validation_labels) / 10.0
            error += (1 - svm.accuracy(train_data, train_labels)) / 10.0
        l[accur] = i - 10
        # plt.plot(i,accur,"go")
        # plt.plot(i, 1 - accur, "ro")
        # plt.plot(i, error, "r^")
    # plt.show()
    m = max(l)
    return 10 ** (l[m]), max(l)


def q6_a():
    print collections.Counter(train_labels[:5000])
    opt_n = best_n()
    print(opt_n)
    opt_C, acc = best_C(opt_n)
    print(opt_C)
    print (acc)
    return opt_n, opt_C, acc  ## result was: 1e-6, 0.1, 0.90754


def q6_b(C=1e-9):
    svm = SVM(9, C)
    svm.train(train_data, train_labels, 1e-6, train_data_size)
    plt.clf()
    for i in xrange(10):
        plt.title("weight vector no.{}".format(i))
        plt.imshow(reshape(svm.w[i], (28, 28)), interpolation='nearest')
        plt.show()


# We saw accodring to q6_a the best etha is: 1e-6 and the best C is: 0.1
def q6_c():
    svm = SVM(9, 0.1)
    svm.train(test_data, test_labels, 1e-6, 10000)
    accur = svm.accuracy(test_data, test_labels)
    print (accur)
    return accur


if __name__ == '__main__':
    # q6_a()
    # q6_b()
    # q6_c() # acc = 0.9167
