from hw2 import *
from numpy import *
from matplotlib.pyplot import *
from sklearn.svm import *

def logRun(ssvvmm, vd, vl, l=10**-10, h=10**10, q=10):
	acc_arr=[]
	while l <= h:
		ssvvmm.C = l
		ssvvmm.fit(train_data,train_labels)
		scoree = ssvvmm.score(vd,vl)
		acc_arr+=[scoree]
		l *= q
	return acc_arr

def sub_a():	
	ssvvmm = LinearSVC(loss='hinge', fit_intercept=False)
	ssvvmm.fit(train_data,train_labels)
	acc_arr = logRun(ssvvmm, vd = validation_data, vl = validation_labels)
	plot(range(-10,11), acc_arr ,marker='o',color="salmon")
	xlabel("logC values")
	ylabel("Prediction accuracy")
	show()
	print ("best C is 10^" + str(acc_arr.index(max(acc_arr))-10))

	acc_arr = logRun(ssvvmm, vd = train_data, vl = train_labels)
	plot(range(-10,11), acc_arr ,marker='o',color="salmon")
	xlabel("logC values")
	ylabel("Prediction accuracy")
	show()

def sub_c():
	ssvvmm = LinearSVC(loss='hinge', fit_intercept=False)
	ssvvmm.C = 10**-7
	ssvvmm.fit(train_data,train_labels)
	arr = ssvvmm.densify().coef_
	w = arr[0]
	imshow(reshape(w,(28,28)), interpolation = 'nearest')
	show()

def sub_d():
	ssvvmm = LinearSVC(loss='hinge', fit_intercept=False)
	ssvvmm.C = 10**-7
	ssvvmm.fit(train_data,train_labels)
	print(ssvvmm.score(test_data, test_labels))
