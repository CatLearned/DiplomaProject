import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from pylab import arange, fft
from scipy.fftpack import fftfreq, irfft, rfft

CURRENT_CHANNEL = 7

my_data = genfromtxt('C:\\ssvep1_eeg_.csv', delimiter=',', skip_header=True)
my_data = my_data.transpose()
print(my_data)

first = 1000
last = 1280
# last = len(my_data[CURRENT_CHANNEL])

x = arange(last - first)
y = my_data[CURRENT_CHANNEL]-np.average(my_data[CURRENT_CHANNEL])
y = y[first:last]

plt.xlabel('Frequency ($Hz$)')
plt.ylabel('Amplitude ($Unit$)')

Fs = 128
n = len(y)  # length of the signal
k = arange(n)
T = n / Fs

frq = k / T  # two sides frequency range
frq = frq[range(int(n / 2))]  # one side frequency range
Y = fft(y) / n  # fft computing and normalization
Y = Y[range(int(n / 2))]

plt.subplot(2,1,2)
plt.plot(frq, abs(Y), 'r')
plt.subplot(2,1,1)
plt.plot(x, y)
plt.show()




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