from src.Bci.Bci import BciThread
from src.Helper.SsvepPlotter import Plotter
from src.Helper.TestObj import TestObj
from src.Helper.FileWritter import FileWritter
from src.Detectors.FFT import FastFourierTransform
from src.Helper.UdpSocket import SocketUDP
import numpy as np
import matplotlib.pyplot as plt
from pylab import arange

SAMPLE_RATE = 256  # Гц
DURATION = 1  # Секунды

def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate*duration, endpoint=False)
    frequencies = x * freq
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


# Генерируем сигналы, которые длится 1 секунду
y0 = arange(256)                                        # Timestamp
x, y1 = generate_sine_wave(10, SAMPLE_RATE, DURATION)   # TP9
y1 = y1 * 1000
x, y2 = generate_sine_wave(20, SAMPLE_RATE, DURATION)   # AF7
y2 = y2 * 1000
x, y3 = generate_sine_wave(30, SAMPLE_RATE, DURATION)   # AF8
y3 = y3 * 1000
x, y4 = generate_sine_wave(28, SAMPLE_RATE, DURATION)   # TP10
y4 = y4 * 1000
y5 = np.array([0] * 256)                                # Референт
data = np.column_stack((y0, y1, y2, y3, y4, y5))        # Объединение

FFT = FastFourierTransform(SAMPLE_RATE)
FFT.update_data(data)

