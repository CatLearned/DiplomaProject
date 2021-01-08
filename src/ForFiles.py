import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from pylab import arange, fft
from scipy.signal import lfilter, butter

CURRENT_CHANNEL = 11                                                      # Muse (1-5), Other (1 - ???)
FS = 128                                                                 # 256 - MUSE, 128 - Other

my_data = genfromtxt('Files\\ssvep1_eeg_.csv', delimiter=',', skip_header=True)
my_data = my_data.transpose()

b, a = butter(3, 0.9)                                                    # Параметры для фильтра

y_all = my_data[CURRENT_CHANNEL]                                         # Signal (Исходный)
y_all = my_data[CURRENT_CHANNEL] - np.average(my_data[CURRENT_CHANNEL])  # Signal (Усреднённый), для Muse не нужен
y_all = lfilter(b, a, y_all)                                             # Фильтрация сигнала, для Muse не нужна
first = 0


#last = first + FS
last = len(my_data[CURRENT_CHANNEL])

while (True):
    x = arange(last - first)
    #y = my_data[CURRENT_CHANNEL]-np.average(my_data[CURRENT_CHANNEL])
    y = y_all[first:last]

    plt.xlabel('Frequency ($Hz$)')
    plt.ylabel('Amplitude ($Unit$)')


    n = len(y)  # length of the signal
    k = arange(n)
    T = n / FS


    fourier = abs(np.fft.fft(y))
    n = y.size
    timestep = 1/FS
    freq = np.fft.fftfreq(n, d=timestep)

#Ytest = fftfreq(y)
#print (freq)
#print (fourier)

    frq = k / T  # two sides frequency range
    frq = frq[range(int(n / 2))]  # one side frequency range
    Y = fft(y) / n  # fft computing and normalization
    Y = Y[range(int(n / 2))]

    plt.subplot(3, 1, 1)
    plt.plot(x, y)
    plt.subplot(3, 1, 2)
    plt.plot(frq, abs(Y), 'r')
    plt.subplot(3, 1, 3)
    plt.plot(freq, fourier, 'r')
    plt.show()
    first = first + FS
    last = last + FS




#t = np.linspace(0, 2*np.pi, 1000, endpoint=True)
#f = 3.0 # Frequency in Hz
#A = 100.0 # Amplitude in Unit
#s = A * np.sin(2*np.pi*f*t) # Signal
#dt = t[1] - t[0] # Sample Time

#W = fftfreq(s.size, d=dt)
#f_signal = rfft(s)

#cut_f_signal = f_signal.copy()
#cut_f_signal[(np.abs(W)>3)] = 0 # cut signal above 3Hz

#cs = irfft(cut_f_signal)


#plt.plot(s)
#plt.plot(cs)