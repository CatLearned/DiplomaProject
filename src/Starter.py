from src.Bci.Bci import BciThread
from src.Helper.SsvepPlotter import Plotter
from src.Helper.TestObj import TestObj
from src.Helper.FileWritter import FileWritter
from src.Detectors.FFT import FastFourierTransform
from src.Helper.UdpSocket import SocketUDP


#Socket = SocketUDP()
#Plotter_SSVEP = Plotter()
#TestObj1 = TestObj()

#CuFFT.attach(Plotter_SSVEP)
#BCI.attach(TestObj1)
#BCI.attach(Socket)
#BCI.attach(CuFFT)

BCI = BciThread("Main")                     # Инициализация ус-ва
#FileWork = FileWritter("Experiment_8")      # Инициализация блока записи
CuFFT = FastFourierTransform(BCI.get_fs())  # Инициализация модуля обработки сигналов
BCI.attach(CuFFT)                           # Подключение модуля обработки сигналов
BCI.start()                                 # Запуск устройства
x = input()                                 # Ожидание ввода
#print("Начало записи в файл")
#BCI.attach(FileWork)                        # Подключение блока записи к ус-ву
#x = input()                                 # Ожидание ввода
BCI.detach(CuFFT)                           # Отключение МОС
#BCI.detach(FileWork)                        # Отключение записи
BCI.stop()                                  # Остановка ус-ва
BCI.join()                                  # Ожидание завершения работы
print("Конец эксперимента")
