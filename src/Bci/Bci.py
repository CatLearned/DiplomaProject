import threading
from src.Abstract.Abstract_observer import Observer
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time

TIMEOUT = 2
EPOCH_LENGTH = 1
OVERLAP_LENGTH = 0.5#0.8
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH


class BciThread (threading.Thread):
    def __init__(self, name):
        threading.Thread.__init__(self)
        self.__streamsEEG = resolve_byprop('type', 'EEG', timeout=TIMEOUT)
        if len(self.__streamsEEG) == 0:
            raise RuntimeError("Can't find EEG stream.")
        self.__inlet = StreamInlet(self.__streamsEEG[0], max_chunklen=12)
        self.__eeg_time_correction = self.__inlet.time_correction()
        self.__info = self.__inlet.info()
        self.__fs = int(self.__info.nominal_srate())
        self.name = name
        self.__lock = threading.Lock()
        self.__work = False
        self.__observers = set()

    def attach(self, observer: Observer):
        self.__observers.add(observer)

    def event(self, info):
        for observer in self.__observers:
            observer.update_data(info)

    def detach(self, observer: Observer):
        self.__observers.remove(observer)

    def run(self):
        self.__work = True
        while self.__work:
            eeg_data, timestamp = self.__inlet.pull_chunk(timeout=1, max_samples=int(SHIFT_LENGTH * self.__fs))
            info = np.column_stack((timestamp, eeg_data))
            self.event(info)

    def get_fs(self):
        return self.__fs

    def stop(self):
        self.__lock.acquire()
        self.__work = False
        self.__lock.release()
