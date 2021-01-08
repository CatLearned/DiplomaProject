from src.Bci.Bci import BciThread
from src.Helper.SsvepPlotter import Plotter
from src.Helper.TestObj import TestObj
from src.Detectors.Ssvep import FastFourierTransform
from src.Helper.UdpSocket import SocketUDP


BCI = BciThread("Main")
#Socket = SocketUDP()
#Plotter_SSVEP = Plotter()
#TestObj1 = TestObj()
CuFFT = FastFourierTransform(128)#(BCI.get_fs())
#CuFFT.attach(Plotter_SSVEP)
#BCI.attach(TestObj1)
#BCI.attach(Socket)
BCI.attach(CuFFT)
BCI.start()

x = input()

BCI.stop()
BCI.join()
print("Ended")
