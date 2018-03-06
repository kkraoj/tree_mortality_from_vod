import os
import numpy as np

tau = 8.
os.chdir('D:/Krishna/Acads/Q4/ML/HW/spam_data')
def readMatrix(file):
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
    category = (np.array(Y) * 2) - 1
    return matrix, tokens, category

def svm_train(matrix, category):
    state = {}
    M, N = matrix.shape
    #####################
    Y = category
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(matrix.T)
    K = np.exp(-(squared.reshape((1, -1)) + squared.reshape((-1, 1)) - 2 * gram) / (2 * (tau ** 2)) )

    alpha = np.zeros(M)
    alpha_avg = np.zeros(M)
    L = 1. / (64 * M)
    outer_loops = 40

    alpha_avg
    for ii in xrange(outer_loops * M):
        i = int(np.random.rand() * M)
        margin = Y[i] * np.dot(K[i, :], alpha)
        grad = M * L * K[:, i] * alpha[i]
        if (margin < 1):
            grad -=  Y[i] * K[:, i]
        alpha -=  grad / np.sqrt(ii + 1)
        alpha_avg += alpha

    alpha_avg /= (ii + 1) * M

    state['alpha'] = alpha
    state['alpha_avg'] = alpha_avg
    state['Xtrain'] = matrix
    state['Sqtrain'] = squared
    ####################
    return state

def svm_test(matrix, state):
    M, N = matrix.shape
    output = np.zeros(M)
    ###################
    Xtrain = state['Xtrain']
    Sqtrain = state['Sqtrain']
    matrix = 1. * (matrix > 0)
    squared = np.sum(matrix * matrix, axis=1)
    gram = matrix.dot(Xtrain.T)
    K = np.exp(-(squared.reshape((-1, 1)) + Sqtrain.reshape((1, -1)) - 2 * gram) / (2 * (tau ** 2)))
    alpha_avg = state['alpha_avg']
    preds = K.dot(alpha_avg)
    output = np.sign(preds)
    ###################
    return output

def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print 'Error: %1.4f' % error
    return(error)

def part_d():
    train_size=np.array([50,100,200,400,800,1400])
    error=np.zeros(len(train_size))
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    i=0
    for size in train_size:    
        filename='MATRIX.TRAIN.%d'%size
        trainMatrix, tokenlist, trainCategory = readMatrix(filename)
        state = svm_train(trainMatrix, trainCategory)
        output = svm_test(testMatrix, state)
        error[i]=evaluate(output, testCategory)
        i+=1
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(3,3))
    ax.plot(train_size,error,color='k')
    ax.set_xlabel('Training set size')
    ax.set_ylabel('Test error')
    ax.set_title('SVM classifier')
    plt.tight_layout()
    plt.savefig('6d')


def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN.400')
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')

    state = svm_train(trainMatrix, trainCategory)
    output = svm_test(testMatrix, state)

    evaluate(output, testCategory)
    part_d()
    return

if __name__ == '__main__':
    main()
