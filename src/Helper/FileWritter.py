from src.Abstract.Abstract_observer import Observer
import time
import csv


class FileWritter (Observer):
    def __init__(self, N):
        if N is None:
            self.__f = open('Files\\Exp_' + str(time.time()) + '.csv', 'w')
        else:
            self.__f = open('Files\\Exp_' + str(N) + '_time' + str(time.time()) + '.csv', 'w')
        self.__f.write('timestamps,TP9,AF7,AF8,TP10,Right AUX,Marker0')
        self.__writer = csv.writer(self.__f)

    def update_data(self, data):
        self.__writer.writerows(data)
