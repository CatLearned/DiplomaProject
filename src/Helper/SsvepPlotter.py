from src.Abstract.Abstract_observer import Observer
import matplotlib.pyplot as plt
import numpy as np


class Plotter(Observer):
    def __init__(self):
        self.__x = np.arange(128)
        self.__y = [0] * 128

        fig, ax = plt.subplots()
        [line] = ax.step(self.__x, self.__y)

        #self.__fig, self.__ax = plt.subplots(1, 1)

        #self.__x = np.arange(128)
        #self.__y = [0] * 128

        #self.__ax.set_aspect('equal')
        #self.__ax.hold(True)

        #self.__ax.set_title('Mind spectrum')
        #self.__ax.set_xlabel('Frequency (Hz)')
        #self.__ax.set_ylabel('Power')
        #self.__ax.set_ylim(0, 128)
        #self.__ax.set_xlim(0, 55)
        #plt.show(block=False)


    def update_data(self, data):

        for i in range(128):
            self.__y[i] = data[i]

        self.__ax.plot(self.__x, self.__y, 'r')
        self.__fig.canvas.draw_idle()



