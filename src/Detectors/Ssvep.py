from src.Abstract.Abstract_observer import Observer
from pylab import arange, fft
#import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
from numpy import genfromtxt


class FastFourierTransform (Observer):
    def __init__(self, fs):
        self.__observers = set()

        self.__fs = fs
        self.__iteration = 0
        self.__queue1 = deque(maxlen=256)
        for i in range(256):
            self.__queue1.append(0)
        #plt.plot(frq, res1, 'r')

    def update_data(self, data):
        data = data.transpose()
        for x in data[1]:
            self.__queue1.append(x)
        data1 = self.__queue1.copy()
        self.__calculation(data1, 0)

    def __calculation(self, data1, timestamp):
        #print("start calc", time.time())
        #print("Func Data[0]",data[0])
        y1 = data1#data[1]
        #y2 = data[2]
        #y3 = data[3]
        #y4 = data[4]
        #print(data1)
        n = len(data1)
        k = arange(n)
        t = n / self.__fs
        #print(n, k, t)
        frq = k / t
        frq = frq[range(int(n / 2))]
        res1 = fft(y1) / n
        res1 = res1[range(int(n / 2))]
        res1 = abs(res1)
        #res1 = abs(res1)
        #res2 = fft(y2) / n
        #res2 = res2[range(int(n / 2))]
        #res3 = fft(y3) / n
        #res3 = res3[range(int(n / 2))]
        #res4 = fft(y4) / n
        #res4 = res4[range(int(n / 2))]
        #print("Real print")
        res1 = np.around(res1)
        res1 = res1.astype(int)
        print(res1[:52])
        #print(res1[19], res1[20], res1[21], "      ", res1[29], res1[30], res1[31])
        self.event(np.around(res1))
        #print("end calc", time.time())

    def attach(self, observer: Observer):
        self.__observers.add(observer)

    def event(self, info):
        for observer in self.__observers:
            observer.update_data(info)

    def detach(self, observer: Observer):
        self.__observers.remove(observer)

