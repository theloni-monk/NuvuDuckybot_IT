import socket
import pickle
import io
import cv2
import zstd

ip = "18.111.87.85"
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.connect((ip, 444))

def recv(size=1024):
    data = bytearray()
    while 1:
        buffer = s.recv(1024)
        data+=buffer
        if len(buffer)==1024:
            pass
        else:
            return data
        

while 1:
    print("Reading...")
    r = recv()
    if len(r) == 0:
        continue
    print("Read {}KB".format(int(len(r)/1000)))
    
    b = zstd.decompress(r)
    
    print("Done reading...")
    img = pickle.load(b)
    cv2.imshow("feed",img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

