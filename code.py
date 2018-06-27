import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def softmax(x):
   
    k=len(x)
    s=[]
    m=max(x)
    array = [np.exp(x[t] - m) for t in range(k)]
    sum = np.sum(array)
    for i in range(k):
        s.append(np.exp(x[i]-m)/sum)
	
    return s

def sigmoid(x):
  
    
    s=1.0/(np.exp(-x)+1)
   
    return s

def forward_prop(data, labels, params):
    
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

   
    m=len(data)
    z1=np.array(W1.dot(data.T))
    z1+=b1
    a1=sigmoid(z1)
    h=a1
    z2=W2.dot(a1)
    z2+=b2
    y=[softmax(p)  for p in z2.T]
    cost=0
    for d in range(m):

     k = labels[d].tolist().index(1)
     cost-=np.log(y[d][k])

    
    return h, y, cost/m

def backward_prop(data, labels, params):
 
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    
    m = len(data)
    z1 = W1.dot(data.T)
    z1+=b1

    a1 = sigmoid(z1)
    h = a1
    z2 = W2.dot(a1)
    z2+=b2
    y = [softmax(p) for p in z2.T]

    gradW1=np.zeros_like(W1)
    gradb1=np.zeros_like(b1)
    gradW2=np.zeros_like(W2)
    gradb2=np.zeros_like(b2)
    reg=0.0001
    for i in range(len(W1)):
        sharedj=0
        for d in range(m):

            k = labels[d].tolist().index(1)
            pred = y[d][k]
            dlogy = 0
            for p in range(10):

                if k != p:
                    dlogy += -y[d][p] * W2[p][i]
                else:
                    dlogy += (1 - pred) * (W2[p][i])

            sharedj+= dlogy * sigmoid(z1[i, d]) * (1 - sigmoid(z1[i, d]))



            gradW1[i]+=-1/m*sharedj*data[d] #+ 2*reg*abs(W1[i])
    for i in range(len(W2)):
        sharedj = 0
        for d in range(m):

            k = labels[d].tolist().index(1)
            pred = y[d][k]
            dlogy = 0
            for p in range(10):

                if k != p:
                    dlogy += -y[d][p] * W2[p][i]
                else:
                    dlogy += (1 - pred) * (W2[p][i])

            sharedj += dlogy * sigmoid(z1[i, d]) * (1 - sigmoid(z1[i, d]))

            gradW2[i]= -1 / m * sharedj*a1[:,d]#+2*reg*abs(W2[i])
    for i in range(len(b1)):
        difflogpred = np.zeros([m,1])
        for d in range(m):

            k = labels[d].tolist().index(1)
            pred = y[d][k]

            dlogy = 0
            for p in range(10):
             if k != i:
               dlogy+= -y[d][k]*W2[p][i]
             else:
                  dlogy+=  (1-pred ) *W2[p][i]
            difflogpred[d] =  dlogy* sigmoid(z1[i,d]) * (1 - sigmoid(z1[i,d]))
        gradb1[i] = -1 / m * ( difflogpred).sum()

    for i in range(len(b2)):
            difflogpred = np.zeros([m,1])
            for d in range(m):

                k = labels[d].tolist().index(1)
                pred = y[d][ k]
                if k != i:
                    dlogy= -y[d][i]
                else:
                    dlogy= 1-pred
                difflogpred[d]= dlogy
            gradb2[i] = -1 / m * ( difflogpred).sum()
  

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad

def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    learning_rate = 0.8
    params = {}

    epochs=100
    mu, sigma = 0, 1  # mean and standard deviation

    params['W1']=np.array([ np.random.normal(mu, sigma, 784) for k in range(300)])
    params['b1']=np.zeros([300,1])
    params['W2']=np.array([ np.random.normal(mu, sigma, 300) for k in range(10)])
    params['b2']=np.zeros([10,1])
    grad=0
    for a in range(epochs):
      a=np.zeros_like(params['W1'])
      b=np.zeros_like(params['b1'])
      c=np.zeros_like(params['W2'])
      d=np.zeros_like(params['b2'])
      for t in range(m):
        grad=backward_prop(trainData[t:t+1,:],trainLabels[t:t+1,:],params)
        a=grad['W1']*0.7+0.3*a
        b= grad['b1']*0.7+0.3*b
        c= grad['W2']*0.7+0.3*c
        d= grad['b2']*0.7+0.3*d
        params['W1'] -= learning_rate*a
        params['b1'] -= learning_rate*b
        params['W2'] -= learning_rate*c
        params['b2'] -= learning_rate*d
      print(nn_test(devData,devLabels,params))
    

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return accuracy,cost

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels


def main():
    np.random.seed(100)
    trainData, trainLabels = readData('ps4/images_train.csv', 'ps4/labels_train.csv')
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]

    devData = trainData[0:1000,:]
    devLabels = trainLabels[0:1000,:]
    trainData = trainData[1000:,:]
    trainLabels = trainLabels[1000:,:]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('ps4/images_test.csv', 'ps4/labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    #testData = (testData - mean) / std

    #params = nn_train(trainData, trainLabels, devData, devLabels)


    readyForTesting = False
    if readyForTesting:
     accuracy = nn_test(testData, testLabels, params)
     print('Test accuracy: %f' % accuracy)
