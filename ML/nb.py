from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt

def readMatrix(file):
    os.chdir('D:/Krishna/Acads/Q4/ML/HW/spam_data')
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    matrix=matrix.astype(int)
    return matrix, tokens, np.array(Y)

def compute_prior(category):
    prior=dict([[1,np.mean(category)],\
    [0,1-np.mean(category)]])
    return prior

def nb_train(matrix,category):
    K=np.shape(matrix)[1]
    state={0:range(K),1:range(K)}
    for cls in np.unique(category):
        row_indices = np.where(category == cls)[0]
        subset      = matrix[row_indices, :]
        Dr=np.sum(subset)+K
        for k in range(K):  
            Nr=np.sum(subset[:,k])+1
            state[cls][k]=Nr
        state[cls]/=Dr
    return state

def predict(state, inputVector,category):
    P=compute_prior(category)
    P.update((x,np.log(y)) for x,y in prior.items())
    for cls in state.keys():
        for i in range(len(inputVector)):
            P[cls]+=inputVector[i]*np.log(state[cls][i])
    prediction = list(P.keys())[list(P.values()).index(max(list(P.values())))]
    return prediction


def nb_test(testSet,state,category):
	predictions = []
	for i in range(len(testSet)):
		result = predict(state, testSet[i],category)
		predictions.append(result)
	return predictions

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error
    return(error)
    
def token_imp(state,tokenlist):
    n=5
    inds=np.log(state[1]/state[0]).argsort()[::-1][:n]
    print([tokenlist[i] for i in inds])
    
    
def part_c():
    train_size=np.array([50,100,200,400,800,1400])
    error=np.zeros(len(train_size))
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    i=0
    for size in train_size:    
        filename='MATRIX.TRAIN.%d'%size
        trainMatrix, tokenlist, trainCategory = readMatrix(filename)
        state = nb_train(trainMatrix, trainCategory)
        output = nb_test(testMatrix, state, trainCategory)
        error[i]=evaluate(output, testCategory)
        i+=1
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(train_size,error,color='k')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test error')
    ax.set_title('Naive Bayes')
    plt.tight_layout()
    plt.savefig('6c')
    
    
def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state, trainCategory)
    evaluate(output, testCategory)
    token_imp(state,tokenlist)
    part_c()
    return

if __name__ == '__main__':
    main()
