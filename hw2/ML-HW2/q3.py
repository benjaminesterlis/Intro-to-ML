from hw2 import *
from numpy import *
from matplotlib.pyplot import *
from random import *
import math


def dot(a,b):
	assert(len(a)==len(b))
	return sum([a[i]*b[i] for i in xrange(len(a))])

def add(w, c, x): # w = w + c*x
	assert(len(w)==len(x))
	return [w[i] + c*x[i] for i in xrange(len(w))]

def check(w, test, labs):
	assert(len(test) == len(labs))
	count = 0
	for i in xrange(len(test)):
		x, c = test[i], labs[i]
		if (dot(w,x) > 0 and c == 1) or (dot(w,x) <= 0 and c == -1):
			count += 1
	return (count+0.0)/len(test)

def SGD(samp, labs, C, T, u0):
	w = [0]*len(samp[0])
	for t in xrange(1,T+1):
		ut = u0/t
		i = randint(0,len(train_data)-1)
		x = samp[i]
		y = labs[i]
		if y * dot(x, w) < 1:
			w = [(1-ut)*w[k] + ut*C*y*x[k] for k in xrange(len(w))]
	return w

def re(a, b):
	d = (b-a)/19
	return [a+d*i for i in xrange(0,20)]

def sub_a(m=10):
	acc = [0]*10
	for i in xrange(10):
		for j in xrange(m):
			print(i,j)
			w = SGD(train_data, train_labels, 1, 1000, 10**(-10+i))
			acc[i] += check(w, validation_data, validation_labels)
	plot(xrange(-10,0), [k/m for k in acc] ,marker='o',color="salmon")
	xlabel("log(ut) values")
	ylabel("Prediction accuracy")
	show()

def sub_b(m=10):
	u0=10**-6
	arr = [10**i for i in xrange(-10,10)]
	k = 0
	dif = 10
	acc = [0]*20
	for i in arr:
		for j in xrange(m):
			print(i,j,k)
			w = SGD(train_data, train_labels, i, 1000, u0)
			acc[arr.index(i)] += check(w, validation_data, validation_labels)
	max_index = acc.index(max(acc))
	dif = abs(arr[max_index-1] - arr[max_index+1])
	while dif > 0.01:
		acc = [0]*20
		arr = re(arr[max_index-1] , arr[max_index+1])
		for i in arr:
			for j in xrange(m):
				print(i,j,k)
				w = SGD(train_data, train_labels, i, 1000, u0)
				acc[arr.index(i)] += check(w, validation_data, validation_labels)
		max_index = acc.index(max(acc))
		dif = abs(arr[max_index-1] - arr[max_index+1])
		k += 1

	plot(arr, [k/m for k in acc] ,marker='o',color="salmon")
	xlabel("logC values")
	ylabel("Prediction accuracy")
	show()

def sub_c():
	c = 1
	u0 = 10**-6
	T = 20000
	w = SGD(train_data, train_labels, c, T, u0)
	imshow(reshape(w,(28,28)), interpolation = 'nearest')
	show()

def sub_d():
	c = 1
	u0 = 10**-6
	T = 20000
	w = SGD(train_data, train_labels, c, T, u0)
	return check(w, test_data, test_labels)

sub_b(2)