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
        self.__queue2 = deque(maxlen=256)
        self.__queue3 = deque(maxlen=256)
        self.__queue4 = deque(maxlen=256)
        self.__queueRef = deque(maxlen=256)
        for i in range(256):
            self.__queue1.append(0)
            self.__queue2.append(0)
            self.__queue3.append(0)
            self.__queue4.append(0)
            self.__queueRef.append(0)

    def update_data(self, data):
        data = data.transpose()
        for x in data[1]:
            self.__queue1.append(x)
        for x in data[2]:
            self.__queue2.append(x)
        for x in data[3]:
            self.__queue3.append(x)
        for x in data[4]:
            self.__queue4.append(x)
        for x in data[5]:
            self.__queueRef.append(x)
        data1 = np.array(self.__queue1.copy())
        data2 = np.array(self.__queue2.copy())
        data3 = np.array(self.__queue3.copy())
        data4 = np.array(self.__queue4.copy())
        dataRef = np.array(self.__queueRef.copy())
        data1 = data1 - dataRef
        data2 = data2 - dataRef
        data3 = data3 - dataRef
        data4 = data4 - dataRef
        dataAll = (data1 + data2 + data3 + data4) / 4
        self.__calculation(dataAll, 0)

    def __calculation(self, dataAll, timestamp):

        n = dataAll.size
        fourier_y = abs(np.fft.fft(dataAll) / n)
        fourier_y = fourier_y[range(int(n / 4))]
        res = fourier_y
        for i in range(64):
            print(i, res[i])

        #print(np.around(fourier_y))
        self.event(np.around(fourier_y))

    def attach(self, observer: Observer):
        self.__observers.add(observer)

    def event(self, info):
        for observer in self.__observers:
            observer.update_data(info)

    def detach(self, observer: Observer):
        self.__observers.remove(observer)

