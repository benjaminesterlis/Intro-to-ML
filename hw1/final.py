import numpy.random
from sklearn.datasets import fetch_mldata
from numpy import linalg as LA
import numpy as np
import matplotlib.pyplot as plt

mnist = fetch_mldata('MNIST original')
data = mnist['data']
labels = mnist['target']
idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]

def accuracy_for_k(k):
    j = 0
    acc = 0
    for img in test:
         i = K_NNeighbor(train[:1000] ,train_labels ,img ,k)
         if ( i == test_labels[j]):
             acc += 1
         j += 1
    return ((acc+0.0)/j)

def accuracy_for_n():
    acc = [ 0 for i in range(50)]
    nums = [i*100 for i in range(1,51)]
    for n in nums:
        print(n)
        j = 0
        accu = 0
        for img in test:
            i = K_NNeighbor(train[:n] ,train_labels ,img ,1)
            if ( i == test_labels[j]):
                accu += 1
            j += 1
        acc[n//100-1] = (accu+0.0)/j
    print(acc)
    return acc
            

def accuracy_for_range():
    acc = [ 0 for i in range(100)]
    j = 0
    for img in test:
        i = NNeighbors(train[:1000] ,train_labels ,img)
        for k in range(100):
            if(i[k] == test_labels[j]):
                acc[k] += 1
        j += 1
    return [(accur+0.0)/j for accur in acc]

def K_NNeighbor(ImageSet,labelsVector,QImg ,k):
    #if True:
    #   return random.randint(0,10) #gives 0.1 accu = 1/10

    IS = np.array(ImageSet)
    #QI = np.array([QImg for i in range(len(ImageSet))])
    #I = np.subtract(IS,QI)
    indexlist = np.argsort(LA.norm(IS-QImg,axis=1))
    k_indeces = indexlist[:k]

    accu = [0 for i in range(10)]
    for i in k_indeces:
        accu[(int)(labelsVector[i])] += 1
    maxind = 0
    for i in range(1,10):
        if(accu[i]>accu[maxind]):
            maxind = i
    return maxind

def NNeighbors(ImageSet,labelsVector,QImg):
    #if True:
    #   return random.randint(0,10) #gives 0.1 accu = 1/10

    label = 0
    IS = np.array(ImageSet)
    QI = np.array([QImg for i in range(len(ImageSet))])
    I = np.subtract(IS,QI)
    indexlist = np.argsort(LA.norm(I,axis=1))
    maxindeces = [ 0 for i in range(100)]
    for k in range(1,101):
        k_indeces = indexlist[:k]
        accu = [0 for i in range(10)]
        for i in k_indeces:
            accu[(int)(labelsVector[i])] += 1
        maxind = 0
        for i in range(1,10):
            if(accu[i]>accu[maxind]):
                maxind = i
        maxindeces[k-1] = maxind
    return maxindeces

if __name__ == '__main__':
    plt.plot(accuracy_for_n())
    plt.ylabel('accuracy')
    plt.savefig("n_accu.png")
    plt.show()


