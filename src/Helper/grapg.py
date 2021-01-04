import matplotlib.pyplot as plt
import numpy as np
import time
import collections

fig = plt.figure()
ax = fig.add_subplot(111)

# some X and Y data
x = np.arange(10000)
y = np.random.randn(10000)
array = collections.deque([None] * 1000, maxlen=1000)

li, = ax.plot(x, array)

# draw and show it
ax.relim()
ax.autoscale_view(True,True,True)
fig.canvas.draw()
plt.show(block=False)

# loop to update the data
while True:
    try:
        #y[:-10] = y[10:]
        #y[-10:] = np.random.randn(10)
        array.append(y[:-10])
        # set the new data
        #li.set_ydata(y)

        fig.canvas.draw()

        time.sleep(0.1)
    except KeyboardInterrupt:
        break