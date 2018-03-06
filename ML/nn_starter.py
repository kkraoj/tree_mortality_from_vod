import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('D:/Krishna/Acads/Q4/ML/HW/')

def readData(images_file, labels_file):
    # x = np.loadtxt(images_file, delimiter=',')
    # y = np.loadtxt(labels_file, delimiter=',')
    # return x,y
    x = pd.read_csv(images_file, delimiter=',', header=None)
    y = pd.read_csv(labels_file, delimiter=',', header=None)
    return x.values, y.values.flatten()

def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
	### YOUR CODE HERE
    x-=np.max(x)
    nr = np.exp(x)
    s = nr / np.sum(nr, axis=1, keepdims=True)
	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    ### YOUR CODE HERE
    s=1/(1+np.exp(-x))
    ### END YOUR CODE
    return s

def plot_cost(train_cost,dev_cost,filename='ps4p1a1',reg=False):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(train_cost,'k--',label='Training set cost')
    ax.plot(dev_cost,'k-',label='Dev set cost')
    ax.set_xlabel('No. of Epochs')
    ax.set_ylabel('Cost')
    ax.set_title('Cost'+reg*', with regularization')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)
    
def plot_accuraccy(train,dev,filename='ps4p1a2',reg=False):
    plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(4,4))
    ax.plot(train,'k--',label='Training accuracy')
    ax.plot(dev,'k-',label='Dev accuracy')
    ax.set_xlabel('No. of Epochs')
    ax.set_ylabel('Accuraccy')
    ax.set_title('Accuraccy'+reg*', with regularization')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename)

def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    # Forward propagation
    z1 = data.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    y_hat = softmax(z2)
    cost =  -np.sum(labels*np.log(y_hat))/len(data)
    ### END YOUR CODE
    return a1, y_hat, cost

def backward_prop(data, labels, params,reg):
    """
    return gradient of parameters
    """
    lamda=0.0001
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    ### YOUR CODE HERE
    a1, y_hat, cost= forward_prop(data, labels, params)
    delta3 = y_hat-labels
#    delta3[range(num_examples), y] -= 1
    dW2 = (a1.T).dot(delta3)/len(data)
    db2 = np.sum(delta3, axis=0, keepdims=True)/len(data)
    delta2 = delta3.dot(W2.T) * a1*(1-a1)
    dW1 = np.dot(data.T, delta2)/len(data)
    db1 = np.sum(delta2, axis=0)/len(data)
    
    if reg:
        dW2 += lamda *2* W2
        dW1 += lamda *2* W1
    ### END YOUR CODE

    grad = {}
    grad['W1'] = dW1
    grad['W2'] = dW2
    grad['b1'] = db1
    grad['b2'] = db2

    return grad

   
   
def nn_train(trainData, trainLabels, devData, devLabels,reg=False):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 5
    params = {}
    num_epoch=30
    num_examples=1000
    K=np.shape(trainLabels)[1]
    ### YOUR CODE HERE
    #initialize
    W1 = np.random.randn(n, num_hidden) 
    b1 = np.zeros((1, num_hidden))
    W2 = np.random.randn(num_hidden, K) 
    b2 = np.zeros((1, K))
    params = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    train_cost, dev_cost, train_acc, dev_acc=\
            np.empty(num_epoch),np.empty(num_epoch), \
            np.empty(num_epoch),np.empty(num_epoch)
    for i in range(num_epoch):
        for j in np.arange(0,m,step=num_examples):
            data,labels=trainData[j:j+num_examples,:],\
                                 trainLabels[j:j+num_examples]
            grad=backward_prop(data,labels, params,reg)
            
            #gradient descent
            W1 += -learning_rate * grad['W1']
            b1 += -learning_rate * grad['b1']
            W2 += -learning_rate * grad['W2']
            b2 += -learning_rate * grad['b2']

            # Assign new parameters to the model
            params = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        a1, y_hat, train_cost[i]= forward_prop(trainData, trainLabels, params)
        train_acc[i]=compute_accuracy(y_hat, trainLabels)
        a1, y_hat, dev_cost[i]= forward_prop(devData, devLabels, params)
        dev_acc[i]=compute_accuracy(y_hat, devLabels)
    filename='ps4_cost'+reg*'_reg'
    plot_cost(train_cost,dev_cost,filename,reg)
    filename='ps4_accuracy'+reg*'_reg'
    plot_accuraccy(train_acc,dev_acc,filename,reg)
    np.save('ps4_params'+reg*'_reg'+'.npy', params) 
    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
#    trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')
#    np.save('trainData.npy',trainData)
#    np.save('trainLabels.npy',trainLabels)
    trainData,trainLabels = \
    np.load('trainData.npy'),np.load('trainLabels.npy')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:10000,:]
    devLabels = trainLabels[0:10000,:]
    trainData = trainData[10000:,:]
    trainLabels = trainLabels[10000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std
    #--------------------------------------------------------
#    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
#    np.save('testData.npy',testData)
#    np.save('testLabels.npy',testLabels)	
    testData,testLabels = \
    np.load('testData.npy'),np.load('testLabels.npy')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    
    #-------------------------------------------------------
    params = nn_train(trainData, trainLabels, devData, devLabels,False)

    readyForTesting = True
    if readyForTesting:
        accuracy = nn_test(testData, testLabels, params)
	print 'Test accuracy: %f' % accuracy

if __name__ == '__main__':
    main()
