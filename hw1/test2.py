import numpy.random
from sklearn.datasets import fetch_mldata
from ex1_all_k import K_NNeighbor

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]



print('this is the data size:',len(idx))
acc = [ 0 for i in range(100)]
j = 0
for img in test:
     i = K_NNeighbor(train[:1000] ,train_labels ,img)
     for k in range(100):
         if(i[k] == test_labels[j]):
             acc[k] += 1
     j += 1
print([(accur+0.0)/j for accur in acc])
