from src.Abstract.Abstract_observer import Observer

FREQ_STOP = 5       # Частота стимула "Стоп"
N_STOP = 10         # Порог стимула "Стоп"
FREQ_FOR = 10       # Частота стимула "Вперёд"
N_FOR = 8           # Порог стимула "Вперёд"
FREQ_BACK = 15
N_BACK = 6
FREQ_RIGHT = 20
N_RIGHT = 5
FREQ_LEFT = 25
N_LEFT = 4


class Detector(Observer):
    def update_data(self, data):
        if data[FREQ_STOP - 1] > N_STOP:  # Проверка преодоления порога "Стоп"
            print("Stop")
        elif data[FREQ_FOR - 1] > N_FOR:  # Проверка преодоления порога "Вперёд"
            print("Forward")
        elif data[FREQ_BACK - 1] > N_BACK:
            print("Back")
        elif data[FREQ_RIGHT - 1] > N_RIGHT:
            print("Right")
        elif data[FREQ_LEFT - 1] > N_LEFT:
            print("Left")
