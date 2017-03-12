import numpy.random
from sklearn.datasets import fetch_mldata
from ex1 import K_NNeighbor

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :]
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :] #change nekoodotaim
test_labels = labels[idx[10000:]]

print('this is the data size:',len(idx))
j = 0
acc =0
for img in test:
     i = K_NNeighbor(train[:1000] ,train_labels ,img ,10)
     if ( i == test_labels[j]):
         acc += 1
     j += 1
print((acc+0.0)/j)
