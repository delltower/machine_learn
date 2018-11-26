#coding: utf-8
import numpy as np
import time
import sys

def loadData(dataFile):
    data = []
    label = []
    with open(dataFile, "r") as fp:
        for line in fp:
            arr = line.strip().split(',')
            if len(arr) == 5:
                if arr[4] == 'Iris-virginica':
                    dataSub = []
                    dataSub.append(float(arr[0]))
                    dataSub.append(float(arr[2]))
                    data.append(dataSub)
                    label.append(1)
                elif arr[4] == 'Iris-setosa':
                    dataSub = []
                    dataSub.append(float(arr[0]))
                    dataSub.append(float(arr[2]))
                    data.append(dataSub)
                    label.append(-1)
            if len(data) >= 100:
                break
    return np.array(data), np.array(label)

def plotResult(data, labels, weights):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    makers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(labels))])

    x1_min, x1_max = data[:,0].min() -1, data[:,0].max() + 1
    x2_min, x2_max = data[:,1].min() -1, data[:,1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    data2 = np.array([xx1.ravel(), xx2.ravel()]) 
    dataMat = np.c_[np.ones((len(data2.T),1)),np.mat(data2.T)]
    predictions = np.dot(dataMat, weights)
    predictions.reshape(xx1.shape)
    print xx1.shape
    print xx2.shape
    plt.contourf(xx1, xx2, predictions, alpha = 0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2,max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=data[y == cl, 0], y=data[y == cl, 1],
                alpha = 0.8, c = cmap(idx),
                marker = markers[idx], label = cl)
    '''
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    n = len(data)
    for i in range(n):
        if labels[i] == 1:
            xcord1.append(data[i][0]) 
            ycord1.append(data[i][1])
        else:
            xcord2.append(data[i][0]) 
            ycord2.append(data[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='blue', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='red', marker='s')

    plt.show()
    '''
#plot one with weights
def plotTest(data,labels, weights):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    #plot weiths line
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.xlim(-10.0, 10.0)
    plt.ylim(-10.0, 10.0)

    x = [-4.0, 8.0]
    y = []
    for item in x:
        y.append((item * -1 * weights[1][0] - weights[0][0]) / weights[2][0])
    print x
    print y
    ax.add_line(Line2D(x, y,linewidth=2, color='red'))

    #plot all data
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    n = len(data)
    for i in range(n):
        if labels[i] == 1:
            xcord1.append(data[i][0]) 
            ycord1.append(data[i][1])
        else:
            xcord2.append(data[i][0]) 
            ycord2.append(data[i][1])

    ax.scatter(xcord1, ycord1, s=30, c='blue', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='red', marker='s')

    plt.show()

#plot one with weights
def plotOneResults(data, labels, weights):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    #plot all data
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    n = len(data)
    for i in range(n):
        if labels[i] == 1:
            xcord1.append(data[i][0]) 
            ycord1.append(data[i][1])
        else:
            xcord2.append(data[i][0]) 
            ycord2.append(data[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='blue', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='red', marker='s')
    '''
    #plot weights
    x = np.arange(0.0, 10.0, 0.1)
    y = []
    for item in x:
        a = (-1.0 * weights[2][0] - weights[0][0] * item) / weights[1][0]
        y.append(a)
    ax.plot(x, y, c = 'black')
    '''
    plt.xlabel('X1'); plt.ylabel('Y');
    x = [0.40, 0.8]
    y = []
    for item in x:
        y.append((item * -1 * weights[1][0] - weights[0][0]) / weights[2][0])
    print x
    print y
    ax.add_line(Line2D(x, y,linewidth=2, color='red'))

    plt.show()

def oldOne(data, label, weights, lr):
    roundNum = 0
    #print weights.shape
    while True:
        num = len(data)
        flag = False
        updateCount = 0
        for i in range(num):
            #count contation
            a = label[i]
            b = np.dot(data[i], weights)
            print "debug"
            print data[i]
            print weights 
            print a*b
            if a * b <= 0 :
                flag = True
                weights = weights + lr * label[i] * data[i].T
                updateCount += 1
            #print weights
            #print weights.shape
        roundNum += 1            
        print "round: %d finish, update count: %d" %(roundNum, updateCount)
        if not flag:
            print "train finish"
            break
    #转换成array类型，有利于绘图    
    return np.array(weights)

def oddOne(data, label,lr):
    roundNum = 0
    #print weights.shape
    #count gram matrix
    gram = np.zeros((len(data), len(data))) 
    m,n = data.shape
    print m,n
    for i in range(m):
        for j in range(m):
            gram[j] = np.dot(np.array(data[i])[0], np.array(data[j])[0])
    print gram
    #weight array
    wa = np.zeros((m,1))
    print wa.shape
    #bias
    b = 0.0
    while True:
        flag = False
        updateCount = 0
        for i in range(m):
            res = 0.0
            for j in range(m):
                res += wa[j] * label[j] * gram[j][i]
            res += b
            res *= label[i]
            #print "index, ", i, res
            if res <= 0 :
                flag = True
                wa[i] += lr
                b += lr * label[i]
                updateCount += 1
            print wa, b
        roundNum += 1            
        print "round: %d finish, update count: %d" %(roundNum, updateCount)
        if not flag:
            print "train finish"
            break
    #count weights
    print wa
    weights = np.zeros((1, len(data[0])))
    for i in range(num):
        weights = weights + wa[i] * label[i] * data[i]
    #转换成array类型，有利于绘图    
    res = []
    res.append(b)
    res.extend(weights)
    print res
    return np.array(res)

def test():
    data, labels = loadData('iris.data')
    #plotResult(data, labels)
    dataMat = np.c_[np.ones((len(data),1)),np.mat(data)]
    labelMat = np.mat(labels).T
    
    weights = np.random.randn(3,1)
    #weights = np.array([-0.97091098,-1.16120145,-0.01755017]).reshape(3,1)
    print "init, ", weights
    start = time.time()
    lr = 0.01
    #weights = oldOne(dataMat, labels, weights, lr)
    weights = oddOne(dataMat, labels, lr)
    print "final, ", weights
    end = time.time()
    print "time : %d" %(end - start)

    #plotOneResults(data, labels, weights)
    plotTest(data, labels, weights)

def test2():
    data, labels = loadData('iris.data')
    #plotResult(data, labels)
    dataMat = np.mat(data)
    labelMat = np.mat(labels).T
    
    weights = np.random.randn(3,1)
    #weights = np.array([-0.97091098,-1.16120145,-0.01755017]).reshape(3,1)
    print "init, ", weights
    start = time.time()
    lr = 0.01
    weights = oddOne(dataMat, labels, lr)
    print "final, ", weights
    end = time.time()
    print "time : %d" %(end - start)

    #plotOneResults(data, labels, weights)
    plotTest(data, labels, weights)


if __name__ == "__main__":
    test2()
