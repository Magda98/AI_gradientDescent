import math
import numpy as np
import matplotlib.pyplot as plt

np.seterr('raise')

def normalization(x, xmin, xmax):
    for idv,val in enumerate(x):
        x[idv] = ((val - xmin)/(xmax-xmin))
    return x

def activation(x):
    B = 1
    return 1/(1 + (np.e**(-B*x)))

def derivative(x):
    return activation(x)*(1-activation(x))

def weights (tab, weight):
    return tab * weight

def hiddenlayer(tab, neuronsNumber, weight):
    arg = []
    ee = []
    # wymiary 'weight' to num x 16, neurnsNumber - ilosc neuronow w warstwie ukrytej 
    for i in range(neuronsNumber):
        e = sum(weights(tab, weight[i]))
        ee.append(e)
        arg.append( activation(e))
    return [arg, ee]

def outlayer(tab, weight):
    x = sum(tab*weight)
    return x

def loss(out, val):
    return val - out

def gradient(l, weight):
    return weight * l

def gradient1(g, weight):
    x =  sum(weight * g)
    return x

def weightUpdate1(weight, localGradient, tab, e, learningRate):
    for i, val in enumerate(weight):
        for j in range(val.size):
            weight[i][j] += learningRate*derivative(e[i])*localGradient[i]*tab[j]
    return weight

def weightUpdate2(weight, l, y, e, learningRate):
    for i in range(weight.size):
        weight[i] += lr*l*derivative(y)*e[i]
    return weight

def neuralNetwork(Pn, Tn, layerNum, neuronsInLayers, epochNum, learningRate, testPn, testTn):
    # wagi dla danej warstwy, 15 - bo tyle jest wejść
    w = []
    w.append(np.random.rand( neuronsInLayers[0], 15))
    for i in range(1, layerNum):
        w.append(np.random.rand(neuronsInLayers[i], neuronsInLayers[i-1] ))
    # dla ostatniej warstwy
    lastl = 0
    w.append(np.random.rand(neuronsInLayers[-1]))
    std_dev, maxx = [], []
    for j in range(epochNum):
        wynik = []
        ll = []
        for i, tab in enumerate(Pn):
            e = []
            ee = []
            ddd = []
            e.append(tab)
            for k in range(layerNum):
                ddd = hiddenlayer(e[k], neuronsInLayers[k], w[k])
                e.append(ddd[0])
                ee.append(ddd[1])

            y = outlayer(e[-1], w[-1])
            ee.append(sum(e[-1] * w[-1]))
            e.append(y)
            l = loss(y, Tn[i])
            # if(l > lastl):
            #     learningRate -= 0.0005
            # else:
            #     learningRate += 0.0005
            # lastl = l
            wynik.append(y)
            ll.append(l)
            wg =  w[::-1]
            wl = neuronsInLayers[::-1]
            g = []
            g.append(gradient(l, wg[0]))
            for k in range(1, layerNum):
                t = wg[k]
                t = t.transpose()
                gg = []
                for p in range(wl[k]):
                    gg.append(gradient1(g[k-1], t[p]))
                g.append(np.asarray(gg))
            g = g[::-1]
            for k in range(layerNum):
                w[k] = weightUpdate1(w[k], g[k], e[k], ee[k], learningRate)
            w[layerNum] = weightUpdate2(w[layerNum], l, ee[-1], e[-2], learningRate)

        mx = np.array(ll).max()
        sd = np.array(ll).std()

        std_dev.append(sd)
        maxx.append(mx)
        print(f'Epoka #{j:02d} (MX: {mx:.10f}, SD: {sd:.10f})', end='\r')
    ww = []
    lll = []
    for i, tab in enumerate(testPn):
        e = []
        ee = []
        ddd = []
        e.append(tab)
        for k in range(layerNum):
            ddd = hiddenlayer(e[k], neuronsInLayers[k], w[k])
            e.append(ddd[0])
            ee.append(ddd[1])

        y = outlayer(e[-1], w[-1])
        ee.append(sum(e[-1] * w[-1]))
        e.append(y)
        l = loss(y, Tn[i])
        ww.append(y)
        lll.append(l)   
    plt.plot(ww)
    plt.plot(testTn)



# wczytanie i przygotowanie danych
testData = []
data = []
with open("zoo.txt") as f:
    data = [list(map(float, x.strip().split(',')[1:])) for x in f]

data = np.array(data).reshape(101,17)
testData = data[0:30,:]
data = data[30:101, :]
data=data[np.argsort(data[:,16])]
testData =testData[np.argsort(testData[:,16])]
data = data.transpose()
testData = testData.transpose()
testPn = testData[0:15]
testTn = testData[16:17][0]
Pn = data[0:15]
Tn = data[16:17][0]
for x, val in enumerate(Pn):
    Pn[x]  = normalization(val, np.min(val), np.max(val))
for x, val in enumerate(testPn):
    testPn[x]  = normalization(val, np.min(val), np.max(val))


lr = 0.5
Pn = Pn.transpose()
testPn = testPn.transpose()


n = [40, 15]
neuralNetwork(Pn, Tn, len(n), n, 20, lr, testPn, testTn)
plt.show()

