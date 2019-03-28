import math
import numpy as np
from random import sample, shuffle
import matplotlib.pyplot as plt
import pickle
from numba import vectorize, float64, int32, int64, float32

@vectorize([int32(int32, int32),
            int64(int64, int64),
            float32(float32, float32),
            float64(float64, float64)])
def ff(a, b):
    return a * b


def normalization(x, xmin, xmax):
    for idv,val in enumerate(x):
        x[idv] = ((val - xmin)/(xmax-xmin))
    return x


def activation(x):
    return 1/(1 + np.e**(-x))

def derivative(x):
    z = activation(x)
    return z*(1 - z)

def weights (tab, weight):
    return ff(tab ,weight)

def hiddenLayer(tab, neuronsNumber, weight):
    arg = []
    ee = []
    for i in range(neuronsNumber):
        e = sum(weights(tab, weight[i]))
        ee.append(e)
        arg.append( activation(e))
    return [arg, ee]

def outLayer(tab, weight):
    tab = np.asarray(tab)
    x = sum(tab*weight)
    return x

def loss(out, val):
    return (val - out)

def error_l(l, weight):
    return weight * l

def error_a(err, weight):
    x =  sum(weight * err)
    return x

def weightUpdate_l(weight, errors, arg, fe, learningRate):
    for i, val in enumerate(weight):
        for j in range(val.size):
            weight[i][j] += learningRate*derivative(fe[i])*errors[i]*arg[j]
    return weight

def weightUpdate_a(weight, ls, arg, out, learningRate):
    for i in range(weight.size):
        weight[i] += learningRate*ls*1*arg[i]
    return weight

def saveModel(wages, neuronsInLayers, layerNum, path):
    with open(path, 'wb') as f:
        pickle.dump(wages, f)
        pickle.dump(neuronsInLayers, f)
        pickle.dump(layerNum, f)

def loadModel(path):
    wages = []
    nauronsInLayers = []
    with open(path, 'rb') as f:
        wages = pickle.load(f)
        nauronsInLayers = pickle.load(f)
        layerNum = pickle.load(f)
    return [wages, nauronsInLayers, layerNum]

def testNet(wages, testPn, testTn, neuronsInLayers, layerNum):
    mse = []
    testResult = []
    for i, tab in enumerate(testPn):
        fe = []
        arg = []
        fe_arg = []
        fe.append(tab)
        for k in range(layerNum):
            fe_arg = hiddenLayer(fe[k], neuronsInLayers[k], wages[k])
            fe.append(fe_arg[0])
            arg.append(fe_arg[1])
        y = outLayer(fe[-1], wages[-1])
        testResult.append(y)
        arg.append(sum(fe[-1] * wages[-1]))
        fe.append(y)
        ls = loss(y, testTn[i])
        mse.append(0.5*ls**2)
    return [np.array(mse).max(), testResult]

def neuralNetwork(Pn, Tn, layerNum, neuronsInLayers, epochNum, learningRate, testPn, testTn):
    # wagi dla danej warstwy, 15 - bo tyle jest wejść
    wages = []
    wages.append(np.random.rand( neuronsInLayers[0], 15))
    for i in range(1, layerNum):
        wages.append(np.random.rand(neuronsInLayers[i], neuronsInLayers[i-1] ))
    # dla ostatniej warstwy
    wages.append(np.random.rand(neuronsInLayers[-1]))

    for j in range(epochNum):
        result = []
        for i, inData in enumerate(Pn):
            fe = []
            arg = []
            fe_arg = []
            fe.append(inData)
            for k in range(layerNum):
                fe_arg = hiddenLayer(fe[k], neuronsInLayers[k], wages[k])
                fe.append(fe_arg[0])
                arg.append(fe_arg[1])
            output = outLayer(fe[-1], wages[-1])
            arg.append(sum(fe[-1] * wages[-1]))
            fe.append(output)
            ls = loss(output, Tn[i])
            result.append(output)

            wage_fl =  wages[::-1]
            nil_fl = neuronsInLayers[::-1]
            errors = []
            errors.append(error_l(ls, wage_fl[0]))
            for k in range(1, layerNum):
                temp = wage_fl[k]
                temp = temp.transpose()
                temp_errors = []
                for p in range(nil_fl[k]):
                    temp_errors.append(error_a(errors[k-1], temp[p]))
                errors.append(np.asarray(temp_errors))
            errors = errors[::-1]
            for k in range(layerNum):
                wages[k] = weightUpdate_l(wages[k], errors[k], fe[k], arg[k], learningRate)
            wages[layerNum] = weightUpdate_a(wages[layerNum], ls, fe[-2], arg[-1], learningRate)

        mse = testNet(wages, testPn, testTn, neuronsInLayers, layerNum)[0]
        if(mse < 0.3):
            break
        print(f'Epoka #{j:02d} mse: {mse:.10f}', end='\r')
    testResult = testNet(wages, testPn, testTn, neuronsInLayers, layerNum)[1]
    saveModel(wages, neuronsInLayers, layerNum, "model")
    plt.plot(result)
    plt.plot(Tn)
    plt.figure()
    plt.plot(testResult)
    plt.plot(testTn)

#main#

# wczytanie i przygotowanie danych
testData = []
data = []
with open("zoo.txt") as f:
    data = [list(map(float, x.strip().split(',')[1:])) for x in f]

data = np.array(data).reshape(101,17)
data = data.tolist()
shuffle(data)
data = np.asarray(data)
testData = data[0:20, :]
data = data[20:101, :]
data=data[np.argsort(data[:,16])]
testData = np.asarray(testData)
testData =testData[np.argsort(testData[:,16])]
data = data.transpose()
testData = testData.transpose()

Pn = data[0:15]
Tn = data[16:17][0]

testPn = testData[0:15]
testTn = testData[16:17][0]

# for x, val in enumerate(Pn):
Pn[12]  = normalization(Pn[12], np.min(Pn[12]), np.max(Pn[12]))
# for x, val in enumerate(testPn):
testPn[12]  = normalization(testPn[12], np.min(testPn[13]), np.max(testPn[13]))

lr = 0.05
Pn = Pn.transpose()
testPn = testPn.transpose()

n = [26, 12, 4]
neuralNetwork(Pn, Tn, len(n), n, 20000, lr, testPn, testTn)

model = loadModel("model1")
result = testNet(model[0], testPn, testTn, model[1], model[2])

# plt.plot(result[1])
# plt.plot(testTn)
plt.show()