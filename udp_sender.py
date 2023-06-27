import socket


class UDPSender:
    def __init__(self, ip="127.0.0.1", port=5053):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg):
        self.sock.sendto(str.encode(msg), (self.ip, self.port))

