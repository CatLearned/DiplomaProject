from src.Bci.Bci import BciThread
from src.Helper.SsvepPlotter import Plotter
from src.Helper.TestObj import TestObj
from src.Helper.FileWritter import FileWritter
from src.Detectors.Ssvep import FastFourierTransform
from src.Helper.UdpSocket import SocketUDP


BCI = BciThread("Main")
FileWork = FileWritter(1)
#Socket = SocketUDP()
#Plotter_SSVEP = Plotter()
#TestObj1 = TestObj()
#CuFFT = FastFourierTransform(256)#(BCI.get_fs())
#CuFFT.attach(Plotter_SSVEP)
#BCI.attach(TestObj1)
#BCI.attach(Socket)
#BCI.attach(CuFFT)
BCI.attach(FileWork)
BCI.start()

x = input()

BCI.stop()
BCI.join()
print("Ended")
