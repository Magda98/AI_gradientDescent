import math
from matplotlib import cm
import numpy as np
from random import sample, shuffle
import matplotlib.pyplot as plt
import pickle
import time
import itertools
from mpl_toolkits.mplot3d import axes3d
import scipy.linalg
from multiprocessing import Pool
from itertools import combinations


def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x, dtype=float)
    for a, (i,j) in zip(m, ij):
        # z = np.add(z, a * x**i * y**j, out=z, casting="unsafe")
        z += a * x**i * y**j
    return z

def job(Pn, Tn,k , epochNum, lr, testPn, testTn):
    czy = 0
    while (czy < 40):
        lr = 0.09
        nn = neuralNetwork(Pn, Tn, 2,k , epochNum, lr, testPn, testTn)
        czy = nn[0]
        if(czy < 40):
            epochNum+= 100
    print(f'end XD')
    return [nn[0],nn[3]]

def task(x):
    z.append(x[0])
    dd.append(x[1])


def normalization(x, xmin, xmax):
    for idv,val in enumerate(x):
        # x[idv] = ((val - xmin)/(xmax-xmin)) #(0,1)
        x[idv] = (2*(val-xmin))/(xmax - xmin) -1
    return x


def activation(x):
    #funkcja aktywacji sigmoidalna unipolarna
    return np.tanh(x)

def derivative(x):
    z = activation(x)
    return 1 - z**2

def weighted_av (tab, weight):
    return tab * weight

def hiddenLayer(tab, neuronsNumber, weight, bias):
    arg = []
    ee = []
    for i in range(neuronsNumber):
        e = sum(weighted_av(tab, weight[i])) + bias[i]
        ee.append(e)
        arg.append( activation(e))
    return [arg, ee]

def outLayer(tab, weight, bias):
    tab = np.asarray(tab)
    x = sum(tab*weight) + bias[0]
    return x

def loss(out, val):
    return (val - out)

def error_l(l, weight, fe):
    err = []
    for k, val in enumerate(fe):
        err.append(l*weight[k] *derivative(val))
    return err

def error_a(err, weight, fe):
    err = np.asarray(err)
    x = derivative(fe)* sum(weight * err)
    return x

def weightUpdate_a(weight, errors, arg, fe, learningRate, bias):
    for i, val in enumerate(weight):
        bias[i] += learningRate*errors[i]
        for j in range(val.size):
            weight[i][j] += learningRate*errors[i]*arg[j]
    return [weight, bias]

def weightUpdate_l(weight, ls, arg, out, learningRate, bias):
    for i in range(weight.size):
        bias[i] += learningRate*ls*1
        weight[i] += learningRate*ls*1*arg[i]
    return [weight,bias]

def saveModel(wages, neuronsInLayers, layerNum, path):
    with open(path, 'wb') as f:
        pickle.dump(wages, f)
        pickle.dump(neuronsInLayers, f)
        pickle.dump(layerNum, f)

def loadModel(path):
    weights = []
    nauronsInLayers = []
    with open(path, 'rb') as f:
        weights = pickle.load(f)
        nauronsInLayers = pickle.load(f)
        layerNum = pickle.load(f)
    return [weights, nauronsInLayers, layerNum]

def testNet(w, testPn, testTn, neuronsInLayers, layerNum, bias):
    pk = 0
    mse = []
    testResult = []
    for i, tab in enumerate(testPn):
        fe = []
        arg = []
        fe_arg = []
        fe.append(tab)
        for k in range(layerNum):
            fe_arg = hiddenLayer(fe[k], neuronsInLayers[k], w[k], bias[k])
            fe.append(fe_arg[0])
            arg.append(fe_arg[1])
        y = outLayer(fe[-1], w[-1], bias[-1])
        testResult.append(y)
        arg.append(sum(fe[-1] * w[-1]))
        fe.append(y)
        ls = loss(y , testTn[i])
        if( abs(loss(y , testTn[i])) <= 0.25 ):
            pk+=1
        mse.append((0.5*(ls**2)))
    pk = pk/(len(testTn)) *100
    return [np.sum(np.array(mse)), testResult, pk]

def initNW(neuronsInLayers, layerNum):
    weights = []
    bias = []
    amin = -1
    amax = 1
    x = 0.5 * (amax - amin)
    y = 0.5 * (amax + amin)
    w_fix = 0.7 * (neuronsInLayers[0] ** (1/15))
    w_rand = (np.random.rand(neuronsInLayers[0], 15) *2 -1)
    w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[0], 1)) * w_rand
    w = w_fix*w_rand
    b = np.array([0]) if neuronsInLayers[0] == 1 else w_fix * np.linspace(-1, 1, neuronsInLayers[0]) * np.sign(w[:, 0])
    x = 0.5 * (amax - amin)
    y = 0.5 * (amax + amin)
    w = x * w
    b = x * b + y
    minmax = np.full((15, 2), np.array([-1, 1]))
    x = 2. / (minmax[:, 1] - minmax[:, 0])
    y = 1. - minmax[:, 1] * x
    w = w * x
    b = np.dot(w, y) + b

    weights.append(w)
    bias.append(b)
    for i in range(1, layerNum):
        w_fix = 0.7 * (neuronsInLayers[i] ** (1/neuronsInLayers[i-1]))
        w_rand = (np.random.rand(neuronsInLayers[i], neuronsInLayers[i-1]) *2 -1)
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[i], 1)) * w_rand
        w = w_fix*w_rand
        b = np.array([0]) if neuronsInLayers[i] == 1 else w_fix * np.linspace(-1, 1, neuronsInLayers[i]) * np.sign(w[:, 0])
        x = 0.5 * (amax - amin)
        y = 0.5 * (amax + amin)
        w = x * w
        b = x * b + y
        minmax = np.full((neuronsInLayers[i-1], 2), np.array([-1, 1]))
        x = 2. / (minmax[:, 1] - minmax[:, 0])
        y = 1. - minmax[:, 1] * x
        w = w * x
        b = np.dot(w, y) + b
        weights.append(w)
        bias.append(b)
    # dla ostatniej warstwy
    # w_fix = 0.7 * (1 ** (1/neuronsInLayers[-1]))
    # w_rand = (np.random.rand(neuronsInLayers[-1], 1) *2 -1)
    # w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[-1], 1)) * w_rand
    # xd = (w_fix*w_rand).flatten()
    weights.append(np.random.rand(neuronsInLayers[-1]))
    b = np.array([0]).astype(float) if 1 == 1 else w_fix * np.linspace(-1, 1, 1) * np.sign(w[:, 0])
    bias.append(b)
    return [weights, bias]

def neuralNetwork(Pn, Tn, layerNum, neuronsInLayers, epochNum, learningRate, testPn, testTn):
    bias = []
    oData = []
    weights = []
    ep = 0
    weights, bias = initNW(neuronsInLayers, layerNum)
    lr_inc = 1.05
    lr_desc = 0.7
    er = 1.04
    last = 0
    for j in range(epochNum):
        result = []
        s_weights = weights
        mse=[]
        for i, inData in enumerate(Pn):
            fe = []
            arg = []
            fe_arg = []
            fe.append(inData)
            for k in range(layerNum):
                fe_arg = hiddenLayer(fe[k], neuronsInLayers[k], weights[k], bias[k])
                fe.append(fe_arg[0])
                arg.append(fe_arg[1])
            output = outLayer(fe[-1], weights[-1], bias[-1])
            arg.append(sum(fe[-1] * weights[-1]))
            ls = loss(output, Tn[i])
            mse.append(0.5*(ls**2))
            result.append(output)
            derFe = arg[::-1]
            wage_fl =  weights[::-1]
            nil_fl = neuronsInLayers[::-1]
            errors = []
            errors.append(error_l(ls, wage_fl[0], derFe[1]))
            for k in range(1, layerNum):
                temp = wage_fl[k]
                temp = temp.transpose()
                temp_errors = []
                dfe = derFe[k+1]
                for p in range(nil_fl[k]):
                    temp_errors.append(error_a(errors[k-1], temp[p], dfe[p]))
                errors.append(np.asarray(temp_errors))
            errors = errors[::-1]
            for k in range(layerNum):
                update = weightUpdate_a(weights[k], errors[k], fe[k], arg[k], learningRate, bias[k])
                weights[k] = update[0]
                bias[k] = update [1]
            update = weightUpdate_l(weights[layerNum], ls, fe[-1], arg[-1], learningRate, bias[-2])
            weights[layerNum] = update[0]
            bias[-2] = update[1]
            bias[-1] += ls


        tData = testNet(weights, testPn, testTn, neuronsInLayers, layerNum, bias)
        # oData.append(tData[0])
        # pk.append(tData[2])
        # lr.append(learningRate)
        # plt.plot(tData[1])
        # plt.plot(testTn)
        # plt.draw()
        # plt.pause(1e-17)
        # plt.clf()
        if(sum(mse) > last*er):
            weights = s_weights
            if(learningRate >= 0.0001):
                learningRate = lr_desc * learningRate
        elif( sum(mse) < last):
            learningRate = lr_inc * learningRate
            if(learningRate > 0.99):
                learningRate = 0.99
        last = sum(mse)
        print(f'Epoka #{j:02d} mse: {tData[0]:.10f}, lr: {learningRate:.4f}, pk: {tData[2]:.2f}%, n: {neuronsInLayers[0]}, {neuronsInLayers[1]}%', end='\r')
        ep = j
    testResult = testNet(weights, testPn, testTn, neuronsInLayers, layerNum,bias)
    # saveModel(weights, neuronsInLayers, layerNum, "model")
    # print(f'end at epoch num: {epochNum}')
    # plt.plot(oData)
    # plt.figure()
    # plt.plot(result)
    # plt.plot(Tn)
    # plt.figure()
    # plt.plot(testResult[1])
    # plt.plot(testTn)
    return [testResult[2], testResult[0], oData, ep]

#main#
if __name__ == "__main__":
    # wczytanie i przygotowanie danych
    testData = []
    data = []
    d=[]
    with open("zoo.txt") as f:
        d = [list(map(float, x.strip().split(',')[1:])) for x in f]

    testData = []
    tab = []
    data = d
    # data = data.tolist()
    shuffle(data)
    data = np.asarray(data)
    data = data.transpose()
    for k in range (16):
        data[k] = normalization(data[k], np.min(data[k]), np.max(data[k]))
    # data[12] = normalization(data[12], np.min(data[12]), np.max(data[12]))
    data = data.transpose()
    tile = [8, 4, 1, 2, 1, 2, 2]
    for k in range(1,8):
        ile = 0
        for m,val in enumerate(data):
            if(val[16] == k):
                ile+=1
                tab.append(m)
                testData.append(val)
            if(ile == tile[k-1]):
                break
    data = np.delete(data, tab, 0)
    # data=data[np.argsort(data[:,16])]
    # testData = data
    testData = np.asarray(testData)
    testData =testData[np.argsort(testData[:,16])]
    data = data.transpose()
    testData = testData.transpose()

    Pn = data[0:15]
    Tn = data[16:17][0]

    testPn = testData[0:15]
    testTn = testData[16:17][0]
    lr = 0.01
    Pn = Pn.transpose()
    testPn = testPn.transpose()
    epochNum = 20000
    neuralNetwork(Pn, Tn, 2,[100,80] , epochNum, lr, testPn, testTn)

    # x = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    # y = [1,2,3,4,5,6,7,8,9,10,11,12]
    # x = np.array(x)
    # y = np.array(y)
    # # dd = []
    # X, Y = np.meshgrid(x,y)
    # XX = X.flatten()
    # YY = Y.flatten()
    # Z=[]
    # # pool = Pool()
    # dd=[];
    # for i, val in enumerate(X):
    #     for j, vall in enumerate(val):
    #         lr = 0.08
    #         k =[]
    #         k.append(vall)
    #         k.append(Y[i, j])
    #         # pool.apply_async(job, args=(Pn, Tn,k , epochNum, lr, testPn, testTn), callback=task)
    #         czy = 0
    #         while (czy < 20):
    #             lr = 0.09
    #             nn = neuralNetwork(Pn, Tn, 2,k , epochNum, lr, testPn, testTn)
    #             czy = nn[0]
    #             if(czy < 20):
    #                 epochNum+= 100
    #         epochNum+= 50;        
    #         Z.append(nn[0])
    #         dd.append(nn[3])
    #     epochNum+=50;
            

    # # pool.close()
    # # pool.join()
    # data = np.c_[XX,YY,Z]
    # f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).transpose()
    # order = 3   # 1: linear, 2: quadratic
    # if order == 1:
    #     # best-fit linear plane
    #     A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    #     C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
        
    #     # evaluate it on grid
    #     # Z = C[0]*X + C[1]*Y + C[2]
    #     # or expressed using matrix/vector product
    #     ZZ = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

    # elif order == 2:
    #     # best-fit quadratic curve
    #     A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    #     C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    #     ZZ = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
    #     # Z = C[4]*X**2. + C[5]*Y**2. + C[3]*X*Y + C[1]*X + C[2]*Y + C[0]
    # elif order == 3:
    #     # best-fit qubic curve
    #      m = polyfit2d(XX,YY,Z)
    #      ZZ = polyval2d(X, Y, m)

    #     # A = np.c_[np.ones(data.shape[0]),data[:,0], data[:,1], data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2, data[:,:2]**3, ]
    #     # C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    #     # Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY,XX**2,YY**2, XX**2*YY, XX*YY**2, XX**3, YY**3], C).reshape(X.shape)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.plot_surface(X, Y, ZZ, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=2, antialiased=True)
    # # ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
    # plt.xlabel('S1')
    # plt.ylabel('S2')
    # ax.set_zlabel('PK[%]')
    # ax.axis('equal')
    # ax.axis('tight')
    # # plt.show()
    # # ax.set_zlim(20, 100)
    # plt.figure()
    # plt.plot(dd)
    # plt.show()

    # model = loadModel("0,05")
    # result = testNet(model[0], testPn, testTn, model[1], model[2])

    # plt.plot(result[1])
    # plt.plot(testTn)
