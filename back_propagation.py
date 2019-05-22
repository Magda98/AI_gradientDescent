import math
from matplotlib import cm
import numpy as np
from random import sample, shuffle
import matplotlib.pyplot as plt
import pickle
import itertools
from mpl_toolkits.mplot3d import axes3d
import scipy.linalg
import time


def normalization(x, xmin, xmax):
    #normalizacja danych
    for idv,val in enumerate(x):
        x[idv] = (2*(val-xmin))/(xmax - xmin) -1
    return x

def prepareData(sort):
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
    if(sort):
        data=data[np.argsort(data[:,16])]
    testData = np.asarray(testData)
    testData =testData[np.argsort(testData[:,16])]
    data = data.transpose()
    testData = testData.transpose()
    return [data, testData]

def activation(x, B = 1):
    #funkcja aktywacji sigmoidalna bipolarna
    return np.tanh(B*x)

def derivative(x, B = 1):
    return 1 -activation(B*x)**2

def weighted_av (tab, weight):
    return tab * weight

def hiddenLayer(tab, neuronsNumber, weight, bias):
    #obliczanie łacznego pobudzenia neuronów w warstwie
    # oraz sygnałow wyjasciowcyh z neuronów
    arg = []
    ee = []
    for i in range(neuronsNumber):
        e = sum(weighted_av(tab, weight[i])) + bias[i]
        ee.append(e)
        arg.append( activation(e))
    return [arg, ee]

def outLayer(tab, weight, bias):
    #obliczanie sygnału wyjściowego w ostatniej warstwie
    tab = np.asarray(tab)
    x = sum(tab*weight) + bias[0]
    return x

def out_error(out, val):
    return (val - out)

def error_l(l, weight, fe):
    #obliczanie delty dla ostatniej wrastwy
    err = []
    for k, val in enumerate(fe):
        err.append(l*weight[k] *derivative(val))
    return err

def error_a(err, weight, fe):
    #obliczanie delty dla pozostałych warstw, procz ostatniej
    err = np.asarray(err)
    x = derivative(fe)* sum(weight * err)
    return x

def weightUpdate_a(weight, errors, arg, fe, learningRate, bias):
    #aktualizacja wag dla wszystkich warstw procz ostatniej
    for i, val in enumerate(weight):
        bias[i] += learningRate*errors[i]
        for j in range(val.size):
            weight[i][j] += learningRate*errors[i]*arg[j]
    return [weight, bias]

def weightUpdate_l(weight, oe, arg, out, learningRate, bias):
    #aktualizacja wag między ostatnią warstwą, a ostatnią ukrytą warstwą
    for i in range(weight.size):
        bias[i] += learningRate*oe*1
        weight[i] += learningRate*oe*1*arg[i]
    return [weight,bias]

def saveModel(wages, neuronsInLayers, layerNum, path):
    #zapisuje dany model sieci w pliku binarnym
    with open(path, 'wb') as f:
        pickle.dump(wages, f)
        pickle.dump(neuronsInLayers, f)
        pickle.dump(layerNum, f)

def loadModel(path):
    #wczystuje dany model sieci
    weights = []
    nauronsInLayers = []
    with open(path, 'rb') as f:
        weights = pickle.load(f)
        nauronsInLayers = pickle.load(f)
        layerNum = pickle.load(f)
    return [weights, nauronsInLayers, layerNum]

def testNet(w, testPn, testTn, neuronsInLayers, layerNum, bias):
    #testuje siec na danych testowych
    pk = 0
    sse = []
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
        oe = out_error(y , testTn[i])
        if( oe**2 <= 0.25 ):
            pk+=1
        sse.append((0.5*(oe**2)))
    pk = pk/(len(testTn)) *100
    return [np.sum(np.array(sse)), testResult, pk]

def initNW(neuronsInLayers, layerNum):
    #inicializacja wag i biasów Nguyen-Widrow'a
    '''
    funkja wzorowana na funkcji z bliblioteki NeuroLab 
    https://pythonhosted.org/neurolab/index.html
    '''
    weights = [] 
    bias = [] #tablica przechowujaca wektory przesuniec
    w_fix = 0.7 * (neuronsInLayers[0] ** (1/15))
    w_rand = (np.random.rand(neuronsInLayers[0], 15) *2 -1)
    w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[0], 1)) * w_rand
    w = w_fix*w_rand
    b = np.array([0]) if neuronsInLayers[0] == 1 else w_fix * np.linspace(-1, 1, neuronsInLayers[0]) * np.sign(w[:, 0])

    weights.append(w)
    bias.append(b)
    for i in range(1, layerNum):
        w_fix = 0.7 * (neuronsInLayers[i] ** (1/neuronsInLayers[i-1]))
        w_rand = (np.random.rand(neuronsInLayers[i], neuronsInLayers[i-1]) *2 -1)
        w_rand = np.sqrt(1. / np.square(w_rand).sum(axis=1).reshape(neuronsInLayers[i], 1)) * w_rand
        w = w_fix*w_rand
        b = np.array([0]) if neuronsInLayers[i] == 1 else w_fix * np.linspace(-1, 1, neuronsInLayers[i]) * np.sign(w[:, 0])
        weights.append(w)
        bias.append(b)
    # dla ostatniej warstwy
    weights.append(np.random.rand(neuronsInLayers[-1]))
    bias.append(np.random.rand(1))
    return [weights, bias]

def delta(arg, weights, neuronsInLayers, oe, layerNum):
    #oblicza deltę przy propagacji wstecznej dla wszystkich warstw
    derFe = arg[::-1] #odwrócona tablica wektorów łącznych pobudzen neuronów z danych warstw
    wage_fl =  weights[::-1] #odwrócona tablica wag
    nil_fl = neuronsInLayers[::-1] #odwrócona tablica z iloscia neuronów w danej warstwie
    d = [] #tablica przechowująca wektory błędów dla danej warstwy
    d.append(error_l(oe, wage_fl[0], derFe[1]))
    for k in range(1, layerNum):
        temp = wage_fl[k]
        temp = temp.transpose()
        temp_d = []
        dfe = derFe[k+1]
        for p in range(nil_fl[k]):
            temp_d.append(error_a(d[k-1], temp[p], dfe[p]))
        d.append(np.asarray(temp_d))
    d = d[::-1]#odwrócenie tablicy błędów
    return d

def neuralNetwork(Pn, Tn, layerNum, neuronsInLayers, epochNum, learningRate, testPn, testTn, lr_inc = 1.05, lr_desc = 0.7, er = 1.04):
    #głowna funkcja odpowiadająca za sieć neuronową
    cost = []
    cost_test = []
    ep = 0
    goal = 0.0002
    weights, bias = initNW(neuronsInLayers, layerNum) # zwracana tablica przechowująca wektory wagowe, przesunięć
    last_cost = 0 #wartosc funkcji kosztu w poprzedniej chwili czasu
    for j in range(epochNum):
        result = [] #tablica przechowująca wyjścia sieci dla danej epoki
        o_weights = weights #przypisanie wag do zmiennej przed wykonaniem się jednej epoki
        o_bias = bias #przypisanie wag do zmiennej przed wykonaniem się jednej epoki
        sse=[]
        for i, inData in enumerate(Pn):
            fe = [] # tablica przechowująca wektory sygnałów wyjściowych z danych warstw
            arg = [] # tablica przechowująca wektory łącznych pobudzen neuronów z danych warstw
            fe_arg = [] #przechowuje listę tablic fe i arg
            fe.append(inData)
            for k in range(layerNum):
                fe_arg = hiddenLayer(fe[k], neuronsInLayers[k], weights[k], bias[k])
                fe.append(fe_arg[0])
                arg.append(fe_arg[1])
            output = outLayer(fe[-1], weights[-1], bias[-1])
            arg.append(sum(fe[-1] * weights[-1]))
            oe = out_error(output, Tn[i])
            sse.append(0.5*(oe**2))
            result.append(output)
            delta_w_b = delta(arg, weights, neuronsInLayers, oe, layerNum)
            for k in range(layerNum):
                update = weightUpdate_a(weights[k], delta_w_b[k], fe[k], arg[k], learningRate, bias[k])
                weights[k] = update[0]
                bias[k] = update [1]
            update = weightUpdate_l(weights[layerNum], oe, fe[-1], arg[-1], learningRate, bias[-2])
            weights[layerNum] = update[0]
            bias[-2] = update[1]
            bias[-1] += oe

        tData = testNet(weights, testPn, testTn, neuronsInLayers, layerNum, bias)
        # pk.append(tData[2])
        # lr.append(learningRate)
        # plt.plot(tData[1])
        # plt.plot(testTn)
        # plt.draw()
        # plt.pause(1e-17)
        # plt.clf()
        sum_sse = sum(sse)
        if( sum_sse > last_cost*er):
            weights = o_weights
            bias = o_bias
            if(learningRate >= 0.0001):
                learningRate = lr_desc * learningRate
        elif( sum_sse < last_cost):
            learningRate = lr_inc * learningRate
            if(learningRate > 0.99):
                learningRate = 0.99
        last_cost = sum_sse
        cost.append(sum_sse)
        cost_test.append(tData[0])
        if (tData[0] < goal):
            ep = j
            break
        print(f'Epoka #{j:02d} sse: {tData[0]:.10f}, lr: {learningRate:.4f}, pk: {tData[2]:.2f}%, er: {er}, n: {neuronsInLayers[0]}, {neuronsInLayers[1]}%', end='\r')
        ep = j
    testResult = testNet(weights, testPn, testTn, neuronsInLayers, layerNum,bias)
    # saveModel(weights, neuronsInLayers, layerNum, "model")
    # print(f'end at epoch num: {epochNum}')
    # plt.figure()
    # plt.plot(result)
    # plt.plot(Tn)
    # plt.figure()
    # plt.plot(testResult[1])
    # plt.plot(testTn)
    return [testResult[2], testResult[0], cost_test, ep, cost, testResult[1]]

#main#
if __name__ == "__main__":
    data, testData = prepareData(False)
    #podział danych na wejściowe i target
    Pn = data[0:15]
    Tn = data[16:17][0]

    testPn = testData[0:15]
    testTn = testData[16:17][0]

    lr = 0.01
    Pn = Pn.transpose()
    testPn = testPn.transpose()
    epochNum = 50
    resultRn = neuralNetwork(Pn, Tn, 2,[26, 15] , epochNum, lr, testPn, testTn)

    # data = data.transpose()
    # data=data[np.argsort(data[:,16])]
    # data = data.transpose()
    # Pn = data[0:15]
    # Tn = data[16:17][0]
    # testPn = testData[0:15]
    # testTn = testData[16:17][0]

    # Pn = Pn.transpose()
    # testPn = testPn.transpose()


    # resultSort = neuralNetwork(Pn, Tn, 2,[50, 15] , epochNum, lr, testPn, testTn)

    # plt.plot(resultRn[2], color = '#4daf4a' ,linewidth=2.0, label='loswa kolejność')
    # plt.plot(resultSort[2], color= '#e55964', linewidth=2.0, label='posortowane')
    # plt.ylabel('koszt')
    # plt.grid(True)
    # plt.xlabel('epoka')
    # plt.legend(loc='upper right')
    # plt.show();
    # plt.plot(resultSort[5], color = '#4daf4a' , marker='o',linewidth=2.0, label='wyjście')
    # plt.plot(testTn, color= '#e55964', marker='o', linewidth=2.0, label='target')
    # plt.ylabel('klasa')
    # plt.grid(True)
    # plt.xlabel('wzorzec')
    # plt.legend(loc='upper left')
    # plt.show()


    ########################################################################################################################################
    # Eksperymenty #
    # S1 = range(1,52,5)
    # S2 = range(1,52,5)
    # S1, S2 = np.meshgrid(S1,S2)
    # PK=[]
    # for i, val in enumerate(S1):
    #     for j, vall in enumerate(val):
    #         model =[]
    #         model.append(vall)
    #         model.append(S2[i, j])
    #         lr = 0.001
    #         result = neuralNetwork(Pn, Tn, 2,model , epochNum, lr, testPn, testTn)     
    #         PK.append(result[0])
                
    # with open(f'test.csv', 'w') as f:
    #     for x,y,z in zip(S1.flatten(), S2.flatten(), PK):
    #         f.write(f'{x};{y};{z}\n')

    # inc = np.arange(1.01,1.09,0.01)
    # desc = np.arange(0.5,0.9,0.05)
    # inc, desc = np.meshgrid(inc,desc)
    # PK=[]
    # for i, row in enumerate(inc):
    #     for j, val in enumerate(row):
    #         result = neuralNetwork(Pn, Tn, 2,[50,15] , epochNum, lr, testPn, testTn, lr_inc = val, lr_desc = desc[i, j])     
    #         PK.append(result[0])


    # e_ratio = np.arange(1.01,1.07,0.005)
    # PK=[]
    # for i, val in enumerate(e_ratio):
    #     nn = neuralNetwork(Pn, Tn, 2,[50,15] , epochNum, lr, testPn, testTn, er = val )     
    #     PK.append(nn[0])

    # with open(f'error_ratio.csv', 'w') as f:
    #         for x,y in zip(e_ratio, PK):
    #             f.write(f'{x};{y}\n')