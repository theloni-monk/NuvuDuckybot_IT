import socket
import numpy as np
import io
import cv2
# zstd might work on other computers but only zstandard will work with mine
import zstandard
import atexit
from netutils import *

class Client:

    def __init__(self, **kwargs):
        self.ip = kwargs.get("serverIp","18.111.87.85")
        self.s = socket.socket()
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.connect((self.ip, kwargs.get("port",444)))
        self.D=zstandard.ZstdDecompressor()
        atexit.register(self.close)

    def recv(self, size=1024):
        print("recv() called")
        data = bytearray()
        while 1:
            buffer = self.s.recv(1024)
            data+=buffer
            if len(buffer)==1024:
                pass
            else:
                print("recv() data ingested")
                return data
        
    def startStream(self):
        img = np.zeros((3,3))
        while True:
            print("Reading...")
            r = recv_msg(self.s)
            if len(r) == 0:
                continue
            print("Read {}KB".format(int(len(r)/1000)))
            print("Done reading...")

            #load decompressed image
            try:
                img = np.load(io.BytesIO(self.D.decompress(r)))
            except Exception as e:
                print(e)

            cv2.imshow("feed",cv2.resize(img, (0,0), fx=3.0, fy=3.0))
            if cv2.waitKey(1) == 27:
                break  # esc to quit
    
    def close(self):
        self.s.close()

if __name__=="__main__":
    client=Client(port=5000)
    client.startStream()
    

