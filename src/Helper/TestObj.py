from src.Abstract.Abstract_observer import Observer
import time


class TestObj(Observer):
    def update_data(self, data):
        print(time.time(), "Info come")
        #x = 0
        #while x < 2:
        #    time.sleep(1)
        #    x = x + 1
        #print(time.time(), "Info come ended")
