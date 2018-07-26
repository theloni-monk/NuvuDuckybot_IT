import socket
import cv2

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('localhost', 8080))

s.listen(10)

while True:
    conn, addr = sock.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    break
    
s.close()
