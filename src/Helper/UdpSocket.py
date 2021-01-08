from src.Abstract.Abstract_observer import Observer
import socket
import time


class SocketUDP (Observer):
    def __init__(self):
        self.__host = 'localhost'
        self.__port = 11111
        self.__address = (self.__host, self.__port)
        self.__udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def stop(self):
        self.__udp_socket.close()

    def update_data(self, data):
        self.__udp_socket.sendto(data, self.__address)


#f = open('ExperimentFakeP300.txt', 'w')

#while True:
#    x = input("Press Enter to send P300 (q + enter - to quit):")
#    ts = time.time()
#    if x == "q":
#        break
#    print(ts)
#    data = b'0x01'
#    udp_socket.sendto(data, address)
#    f.write(str(ts) + '\n')
#    print("Sent P300")

#f.close()
#udp_socket.close()