import numpy as np
import logging as lg
import os
import random

def learn(X):
    """ Learns the fundamental memories represented in the input vector X
    Args:
        Take flattened (1-D) input vectors arrranged in an array
    Returns:
        The interconnection matrix as per the Hebb rule"""
    return (np.dot(X.T, X)-X.shape[0]*np.eye(X.shape[-1]))/X.shape[-1]


def read_data(path):
    """Returns the content from the directory specified by path converting them to
    flattened (1-D) input vectors arranged in an array
    Args:
        Take the directory where the input vectors are as input
    Returns:
        Returns the input vectors flattened and in an array"""
    lg.debug('Veryfying if the data file exists')
    assert os.path.exists(path)
    files=os.listdir(path)
    batch=[]
    i=0
    c=0
    for f in files:
        if not "txt" in f:
            c+=1
            continue ## Ensure that we read only txt files
        fd=open(path+f)
        memory=np.fromstring(fd.read(), dtype=np.int, sep=' ')
        batch.append(memory)
        fd.close()
    arr=np.zeros((len(files)-c,len(batch[1])))
    for i in range(0,len(files)-1): arr[i,:]= batch[i]
    arr[arr == 0] = -1  # Making array bipolar
    lg.info('Converted fundamental memories into a 2d array')
    return arr

def init(path):
    """Initializing a Hopfield network with fundamental memories read from folder"""
    lg.basicConfig(filename='hopfield.log', level=lg.DEBUG, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')

    X = read_data(path)
    np.savetxt('input_file.txt', X, fmt='%d')  ## Use loadtxt with savetxt to load file
    lg.info('Saved input vector in a file')
    W = learn(X)
    np.savetxt('weight_file.txt', W, fmt='%f')
    lg.info('Saved interconnection matrix in a file')
    return

def sgnn(H):
    if H > 0:
        H = 1
    else:
        H = -1
    return H

def retreive(W,d):
    d[d == 0] = -1
    s_old = d
    s_new=np.zeros(s_old.shape)
    flag,t=1,0
    while flag:
        flag=0
        for i in random.sample(range(0,len(d)), len(d)):
            s_new[i]=sgnn(np.dot(W[i,:],s_old))
            if s_new[i] != s_old[i]:
                flag=1;
                t=t+1
                s_old[i] = s_new[i]

    print('Retrived in', t, 'iterations')
    return s_new.astype(int)

def distort(s,d):
    for i in random.sample(range(0,len(s)),d): s[i] = 1-s[i]
    return s


if __name__ == "__main__":
     fname='weight_file.txt'
     path = 'dgt/'
     if not os.path.isfile(fname):
         init(path)
     lg.info('Loading interconnection matrix from the file')
     W = np.loadtxt('weight_file.txt')
     #print W
     f='0.txt'  ## Loading the file whose distoeted image is to presented

     fd = open(path + f)
     memory = np.fromstring(fd.read(), dtype=np.int, sep=' ')
     memory_distorted=distort(memory,30)
     print memory_distorted.reshape(10, 12)
     s_retreived = retreive(W,memory_distorted)
     s_retreived [ s_retreived == -1] = 0
     print s_retreived.reshape(10,12)


