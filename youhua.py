#coding: utf-8
import numpy as np
import time 
import sys

#load data
def loadDataSet(fileName):
    data, labels = [], []
    with open(fileName, "r") as fp:
        for line in fp:
            arr = line.strip().split("\t")
            data.append([float(x) for x in arr[0:-1]]) 
            labels.append(float(arr[-1]))
    #return np.c_[np.ones((len(data),1)),np.mat(data)], np.array(labels)
    return np.array(data), np.array(labels)

#plot one
def plotOne(data, labels):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    n = len(data)
    for i in range(n):
        xcord1.append(data[i]) 
        ycord1.append(labels[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='blue', marker='s')
    plt.xlabel('X1'); plt.ylabel('Y');
    plt.show()

#plot one with weights
def plotOneResults(data, labels, weights, cost):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    n = len(data)
    for i in range(n):
        xcord1.append(data[i]) 
        ycord1.append(labels[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='blue', marker='s')
    x = np.arange(0.0, 2.0, 0.1)
    y = weights[0] + weights[1]*x
    ax.plot(x, y, c = 'red')
    plt.xlabel('X1'); plt.ylabel('Y');
    plt.title("cost=%f" %(cost))
    plt.show()

#plot costs
def plotCost(cost):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord1 = []
    n = len(cost)
    for i in range(n):
        xcord1.append(i) 
        ycord1.append(cost[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xcord1, ycord1, c = 'red')
    plt.xlabel('X1'); plt.ylabel('Y');
    plt.show()

#plot costs
def plotCostArr(costArr, colorArr):
    import matplotlib.pyplot as plt
    xcord1 = []; ycord = []
    n = len(costArr[0])
    for i in range(n):
        xcord1.append(i) 
         
    for m in range(len(costArr)):
        ycord.append([])
        for i in range(n):
            ycord[m].append(costArr[m][i])
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    for m in range(len(costArr)):
        plt.plot(xcord1, ycord[m], c = colorArr[m])
    plt.xlabel('X1'); plt.ylabel('Y');
    plt.show()


#cost count
def calCost(data, labels, weights):
    m = data.shape[0]
    predictions = data.dot(weights)
    cost = (1.0/2*m) * np.sum(np.square(predictions - labels))
    return cost

#
def gradientDescent(data, labels, weights, learnRate=0.01, iterations=100):
    m, n= data.shape[0], data.shape[1]
    costHistory = np.zeros(iterations)
    weightsHistory= np.zeros((iterations, n))
    for i in range(iterations):
        predictions = np.dot(data, weights)
        weights = weights - (1.0/m) * learnRate * (data.T.dot((predictions - labels)))
        weightsHistory[i,:] = weights.T
        costHistory[i] = calCost(data, labels, weights)

    return weights, costHistory, weightsHistory
#
def stocGradientDescent(data, labels, weights, learnRate=0.01, iterations=100, batchSize=20):
    m, n= data.shape[0], data.shape[1]
    print m,n
    costHistory = np.zeros(iterations)
    weightsHistory= np.zeros((iterations, n))
    for i in range(iterations):
        cost = 0.0
        for j in range(batchSize):
            #choose a data compute
            index = np.random.randint(0,m)
            subData = data[index]
            subLabel = labels[index]
            predictions = np.dot(subData, weights) #这是一个向量，但是只有一个数字，计算的只是一个数据的函数值
            weights = weights - (1.0/m) * learnRate * (subData.T.dot((predictions - subLabel)))
            #cost += (1.0/2*m) * np.square((predictions - labels)[0,0])
            cost += calCost(subData, subLabel, weights) 

        weightsHistory[i,:] = weights.T
        costHistory[i] = cost

    return weights, costHistory, weightsHistory

#
def batchGradientDescent(data, labels, weights, learnRate=0.01, iterations=100, batchSize=20):
    m, n= data.shape[0], data.shape[1]
    costHistory = np.zeros(iterations)
    weightsHistory= np.zeros((iterations, n))
    for i in range(iterations):
        cost = 0.0
        #shuffe
        indices = np.random.permutation(m)
        dataShuffe = data[indices]
        labelShuffe = labels[indices]
        for j in range(0,m,batchSize):
            #choose batch data
            subData = dataShuffe[j:j + batchSize]
            subLabel = labelShuffe[j:j + batchSize]
            predictions = np.dot(subData, weights) #这是一个向量
            weights = weights - (1.0/m) * learnRate * (subData.T.dot((predictions - subLabel)))
            cost += calCost(subData, subLabel, weights) 

        weightsHistory[i,:] = weights.T
        costHistory[i] = cost

    return weights, costHistory, weightsHistory
# get step of like netton's method
def getStep(data, labels, oldWeights,incrWeights):
    res = 1.0
    #init range of min and max  
    minLine, maxLine = 0.0,0.0
    x,y,z = 0.0, 0.0, 0.0 
    xCost, yCost, zCost = 0.0, 0.0, 0.0
    h = 0.0001
    #get range of min,max 
    xCost = calCost(data, labels, oldWeights + x * incrWeights)
    y = x + h 
    yCost = calCost(data, labels, oldWeights + y * incrWeights)
    if abs(xCost - yCost) < 0.0001:
        minLine = x
        maxLine = y
    else:
        '''
        print "init x,y"
        print x,y
        print xCost, yCost 
        '''
        if xCost < yCost:
            h = -h
            #swap x,y
            tmp = x; x = y; y = tmp
            tmp = xCost; xCost = yCost; yCost = tmp 
        else:
            h = 2*h
        '''
        print "get x,y"
        print x,y
        print xCost, yCost 
        '''
        z = y + h
        zCost = calCost(data, labels, oldWeights + z * incrWeights)
        count = 0
        while yCost >= zCost:
            count += 1
            if yCost > zCost:
                h *= 2.0
            else:
                h /= 2.0
            #swap x, y
            x = y; y = z
            z = y + h
            xCost = calCost(data, labels, oldWeights + x * incrWeights)
            yCost = calCost(data, labels, oldWeights + y * incrWeights)
            zCost = calCost(data, labels, oldWeights + z * incrWeights)
        '''
            print x,y,z
            print xCost, yCost, zCost
        print "get x, y, z"
        print "count: ", count
        print x,y,z
        print xCost, yCost, zCost
        '''
        minLine = min(x,z)
        maxLine = max(x,z)
    #print "get range: [%.5f, %.5f]"  %(minLine, maxLine)
    #if minLine < 0:
    #    minLine = 0
    #search
    '''
    while maxLine - minLine > 0.0001:
        print "begin search"
        x1 = minLine + (maxLine - minLine) / 4.0
        x2 = maxLine - (maxLine - minLine) / 4.0
        x1Cost = calCost(data, labels, oldWeights + x1 * incrWeights)
        x2Cost = calCost(data, labels, oldWeights + x2 * incrWeights)
        minCost = calCost(data, labels, oldWeights + minLine * incrWeights)
        maxCost = calCost(data, labels, oldWeights + maxLine * incrWeights)
        print minLine, x1, x2,maxLine
        print minCost, x1Cost, x2Cost, maxCost
        if x1Cost < x2Cost:
            maxLine = x2
        else:
            minLine = x1
        print "min,max: %.5f, %.5f" %(minLine, maxLine)
    res = ( minLine + maxLine ) / 2.0      
    '''
    #search by goden split method
    while maxLine - minLine > 0.0001:
        x1 = minLine + 0.382 * (maxLine - minLine)
        x2 = minLine + 0.618 * (maxLine - minLine)
        x1Cost = calCost(data, labels, oldWeights + x1 * incrWeights) 
        x2Cost = calCost(data, labels, oldWeights + x2 * incrWeights) 
        if x1Cost < x2Cost:
            maxLine = x2;
        else:
            minLine = x1
    res = ( minLine + maxLine ) / 2.0
    #print "final step: ", res, calCost(data, labels, oldWeights + res * incrWeights)
    return res

#like newton's method
def likeNewtonMethod(data, labels, weights, iterations=100):
    m, n= data.shape[0], data.shape[1]
    costHistory = np.zeros(iterations)
    weightsHistory= np.zeros((iterations, n))
    #xk = [1,n], this is rotate matrix
    xMat = np.zeros((iterations,n))
    xMat[0] = weights.T
    #Dk = [n_iter,n,n]
    DMat = np.zeros((iterations,n,n))
    DMat[0] = np.eye(n)
    #gk = [n, n_iter], this is rotate matrix
    gMat = np.zeros((iterations, n))
    gMat[0] = (1.0/m) * (data.T.dot((np.dot(data, xMat[0].reshape(1,n).T) - labels))).T
    #dk = [n,n_iter], this is rotate matrix
    dMat = np.zeros((iterations,n))

    for i in range(iterations - 1):
        #print "begin round: ", i
        #count d(i)
        dMat[i] = -1 * np.dot(DMat[i], gMat[i].T).T
        #get step
        step = getStep(data, labels, xMat[i].T ,dMat[i])
        xMat[i+1] = xMat[i].T + step * dMat[i].T
        '''
        print "xk: ",xMat[i],"cost: ",calCost(data, labels, xMat[i].T) 
        print "dk: ",dMat[i]
        print "xk+1: ",xMat[i+1], "cost: ",calCost(data, labels, xMat[i+1].T)
        '''
        #count g(i+1)
        gMat[i+1] =  (1.0/m) * (data.T.dot((np.dot(data, xMat[i+1].reshape(1,n).T)- labels))).T
        #count y(i) s(i)
        y = (gMat[i+1] - gMat[i]).T.reshape(n, 1)
        s = (xMat[i+1] - xMat[i]).T
        #print "y,s: ",y,s
        #count D(i+1)
        DMat[i+1] = DMat[i] 
        DMat[i+1] += (np.dot(s, s.T))/(np.dot(s.T, y)) 
        DMat[i+1] -= np.dot(np.dot(np.dot(DMat[i], y),y.T), DMat[i]) / np.dot(np.dot(y.T, DMat[i]),y) 

        #save result
        weights = xMat[i].T
        costHistory[i] = calCost(data, labels, weights)
        weightsHistory[i,:] = weights.T
        #print "finish %d" %(i)
    return weights, costHistory, weightsHistory

#
def test(lr, n_iter, batchSize = 20):
    print lr, n_iter,batchSize
    data, labels = loadDataSet("one.txt")
    dataMat = np.c_[np.ones((len(data),1)),np.mat(data)]
    labelMat = np.mat(labels).T

    #weights = np.random.randn(2,1)
    weights = np.array([-1.05984693,0.2315732]).reshape(2,1)
    start = time.time()
    weights, costHistory, weightHistory = gradientDescent(dataMat, labelMat, weights, lr, n_iter)
    #weights, costHistory, weightHistory = stocGradientDescent(dataMat, labelMat, weights, lr, n_iter, batchSize)
    #weights, costHistory, weightHistory = batchGradientDescent(dataMat, labelMat, weights, lr, n_iter, batchSize)
    #weights, costHistory, weightHistory = likeNewtonMethod(dataMat, labelMat, weights, n_iter)
    end = time.time()

    print "iters: %d time: %d" %(n_iter, end - start)
    return weights, costHistory, weightHistory 

#
def testDiff(lr, n_iter, batchSize = 20):
    print lr, n_iter,batchSize
    data, labels = loadDataSet("one.txt")
    dataMat = np.c_[np.ones((len(data),1)),np.mat(data)]
    labelMat = np.mat(labels).T
    costArr, colorArr = [], []
    weightold = np.random.randn(2,1)

    weights = weightold
    start = time.time()
    weights, costHistory, weightHistory = gradientDescent(dataMat, labelMat, weights, lr, n_iter)
    end = time.time()
    costArr.append(costHistory)
    colorArr.append('red')
    print "gd iters: %d time: %d color: %s" %(n_iter, end - start, colorArr[-1])
    print costHistory
    '''
    weights = weightold
    start = time.time()
    weights, costHistory, weightHistory = stocGradientDescent(dataMat, labelMat, weights, lr, n_iter)
    end = time.time()
    costArr.append(costHistory)
    colorArr.append('blue')
    print "sgd iters: %d time: %d color: %s" %(n_iter, end - start, colorArr[-1])


    weights = weightold
    start = time.time()
    weights, costHistory, weightHistory = batchGradientDescent(dataMat, labelMat, weights, lr, n_iter, batchSize)
    end = time.time()
    costArr.append(costHistory)
    colorArr.append('green')
    print "bgd iters: %d time: %d color: %s" %(n_iter, end - start, colorArr[-1])
    '''
    weights = weightold
    start = time.time()
    weights, costHistory, weightHistory = likeNewtonMethod(dataMat, labelMat, weights, n_iter)
    end = time.time()
    costArr.append(costHistory)
    colorArr.append('yellow')
    print "like newton iters: %d time: %d color: %s" %(n_iter, end - start, colorArr[-1])
    print costHistory

    plotCostArr(costArr, colorArr)
if __name__ == "__main__":
    test(0.01, int(sys.argv[1]))
    #testDiff(0.01, 50)
