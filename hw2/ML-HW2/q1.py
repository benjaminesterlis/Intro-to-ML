from hw2 import *
from numpy import *
from matplotlib.pyplot import *
from math import log

def dot(a,b):
	assert(len(a)==len(b))
	return sum([a[i]*b[i] for i in xrange(len(a))])

def normalize(vector):
	norma = sqrt(dot(vector,vector))
	if norma == 0:
		return vector
	return [num/norma for num in vector]

def add(w, c, x): # w = w + c*x
	assert(len(w)==len(x))
	return [w[i] + c*x[i] for i in xrange(len(w))]

def perceptron(samples, labs):
	for i in xrange(len(samples)):
		samples[i] = normalize(samples[i])
	w = [0]*len(samples[0])
	for i in xrange(len(samples)):
		x = samples[i]
		c = labs[i]
		prediction = 1 if dot(w,x) >= 0 else -1
		if prediction != c:
			w = add(w,c,x)
	return w

def perm(samp,labs):
	zipped = zip(samp,labs)
	arr = random.permutation(zipped)
	return [a[0] for a in arr], [a[1] for a in arr]

# returns accuracy rate
def check(w, test, labs):
	assert(len(test) == len(labs))
	count = 0
	for i in xrange(len(test)):
		x, c = test[i], labs[i]
		if (dot(w,x) > 0 and c == 1) or (dot(w,x) <= 0 and c == -1):
			count += 1
	return (count+0.0)/len(test)

def sub_a(m=100):
	n = [5, 10, 50, 100, 500, 1000, 5000]
	data = [0]*3*len(n)
	for j in xrange(len(n)):
		samples = train_data[:n[j]]
		labs = train_labels[:n[j]]
		acc = [0]*m
		for i in xrange(m):
			samples, labs = perm(samples,labs)
			w = perceptron(samples, labs)
			print(w)
			acc[i] = check(w, test_data, test_labels)
		acc.sort()
		data[3*j], data[3*j+1], data[3*j+2] = acc[m/20 - 1], acc[m - 1 - m/20], (sum(acc)+0.0)/m
	
	data = reshape(data, (len(n), 3))
	fig, axs = subplots(2,1)
	clust_data = data
	collabel = ("5%", "95%", "mean")
	axs[0].axis('tight')
	axs[0].axis('off')
	the_table = axs[0].table(cellText=clust_data,colLabels=collabel,loc='center')
	axs[1].plot([log(a) for a in n],clust_data[:,1], marker='o')
	show()

def sub_b():
	w = perceptron(train_data, train_labels)
	imshow(reshape(w,(28,28)), interpolation = 'nearest')
	show()

def sub_c():
	w = perceptron(train_data, train_labels)	
	acc = check(w, test_data, test_labels)
	return acc

def sub_d():
	w = perceptron(train_data, train_labels)
	for i in xrange(len(test_data)):
		x, c = test_data[i], test_labels[i]
		if (dot(w,x) > 0 and c == 1) or (dot(w,x) <= 0 and c == -1):
			imshow(reshape(x,(28,28)), interpolation = 'nearest')
			show()
			break
