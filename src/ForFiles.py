import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from pylab import arange
from scipy.signal import lfilter, butter, welch, iirfilter


def butter_bandpass(lowcut, highcut, fs, order=3): # 3 ten sonra lfilter NaN degerler vermeye basliyor
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', analog=False)


def butter_lowpass(cutoff, fs, order=3):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


CURRENT_CHANNEL = 3                                                      # Muse (1-5), Other (9-12) 11 = o2
FS = 128                                                                 # 256 - MUSE, 128 - Epoc

my_data = genfromtxt('Files\\ssvep1_eeg_.csv', delimiter=',', skip_header=True)
#my_data = genfromtxt('Files\\subject1\\ssvep1001_20hz_2_eeg_.csv', delimiter=',', skip_header=True)
#my_data = genfromtxt('Files\\Exp1_time1610304289.3217652.csv', delimiter=',', skip_header=True)
#my_data = genfromtxt('Files\\etalon\\data_2017-09-14-21.20.04.csv', delimiter=',', skip_header=True)
#my_data = genfromtxt('Files\\ssvep\\session010\\recording.csv', delimiter=',', skip_header=True)
my_data = my_data.transpose()

b, a = butter_bandpass(15, 35, FS, 3)
#iirfilter(5, [30/FS, 80/FS], btype='band')
#butter_bandpass(1, 50, FS, 3)
#butter(3, 0.1)                      # Параметры для фильтра


# Переработка
y_ref = my_data[10]
y_1 = my_data[11]# - y_ref
y_2 = my_data[10]# - y_ref
y_3 = my_data[3] - y_ref
y_4 = my_data[4] - y_ref

#y_1 = y_1 - np.average(y_1)

y_all = (y_1) / 1


#y_all = my_data[CURRENT_CHANNEL]                                         # Signal (Исходный)
#y_all = my_data[CURRENT_CHANNEL] - np.average(my_data[CURRENT_CHANNEL])  # Signal (Усреднённый), для Muse не нужен
#y_second = my_data[2]
#y_second = y_second - np.average(y_second)
#y_3 = my_data[3]
#y_3 = y_3 - np.average(y_3)
#y_4 = my_data[4]
#y_4 = y_4 - np.average(y_4)



#y_all = y_all + y_second + y_3 + y_4
#y_all = y_all / 4
#y_all = y_all - np.average(y_all)
#y_ref = my_data[5]
#y_ref = y_ref - np.average(y_ref)
#y_all = y_all - y_ref
#x_all = my_data[20]                                                       # Timestamps
y_all_filtered = lfilter(b, a, y_all)                                    # Фильтрация сигнала, для Muse не нужна

print(np.min(y_1))
print(np.max(y_1))
first = 0
#last = first + 256 #FS
last = len(my_data[CURRENT_CHANNEL])

while (True):
    x = arange(last - first)
    #x = x_all[first:last]
    #y = my_data[CURRENT_CHANNEL]-np.average(my_data[CURRENT_CHANNEL])
    y = y_all[first:last]
    y_filtered = y_all_filtered[first:last]

    n = len(y)  # length of the signal
    k = arange(n)
    T = n / FS

    n = y.size
    fourier_y = abs(np.fft.fft(y) / n)
    fourier_y = fourier_y[range(int(n / 2))]
    f, Pxx_den = welch(y, FS, scaling="spectrum")
    Pxx_den = 10 * np.log10(Pxx_den)
    fourier_y_fil = abs(np.fft.fft(y_filtered) / n)
    fourier_y_fil = fourier_y_fil[range(int(n / 2))]
    f2, Pxx_den2 = welch(y_filtered, FS, scaling="spectrum")
    Pxx_den2 = 10 * np.log10(Pxx_den2)
    timestep = 1/FS
    freq = np.fft.fftfreq(n, d=timestep)
    freq = freq[range(int(n / 2))]
    print(freq)

#Ytest = fftfreq(y)
#print (freq)
#print (fourier)


    plt.subplot(2, 2, 1)
    plt.plot(x, y)
    plt.title("Сигнал без фильтрации")
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')
    plt.subplot(2, 2, 2)
    plt.title("Спектр сигнала без фильтрации")
    plt.ylabel('Амплитуда')
    plt.xlabel('Частота, Гц')
    #plt.plot(f, Pxx_den, 'r')
    plt.plot(freq, fourier_y, 'r')
    plt.subplot(2, 2, 3)
    plt.title("Сигнал с фильтрацией")
    plt.ylabel('Амплитуда')
    plt.xlabel('Время, с')
    plt.plot(x, y_filtered)
    plt.subplot(2, 2, 4)
    plt.title("Спектр сигнала с фильтрацией")
    plt.ylabel('Амплитуда')
    plt.xlabel('Частота, Гц')
    #plt.plot(f2, Pxx_den2, 'r')
    plt.plot(freq, fourier_y_fil, 'r')

    #plt.subplot(3, 1, 3)
    #plt.plot(freq, fourier, 'r')
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