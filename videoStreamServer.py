import socket
import cv2
import camera
import io
import numpy as np
from tempfile import TemporaryFile
import zstandard
import atexit
from netutils import *

class Server:
    def __init__(self,**kwargs):
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('', kwargs.get("port",444)))
        s.listen(10)
        #print("Ready")
        self.s = s
        self.verbose = kwargs.get("verbose",True)
        atexit.register(self.close)

    def serve(self):
        while True:
            self.conn, self.clientAddr = self.s.accept()
            if self.verbose:
                #print('Connected with ' + self.clientAddr[0] + ':' + str(self.clientAddr[1]))
                return
    def startStream(self,getFrame,args=[]):
        #not sure if np.save compression is worth io overhead...
        
        C=zstandard.ZstdCompressor()
        #print("Starting stream...")
        while True:
            Tfile=io.BytesIO()
            #fetch the image
            #print ("Fetching frame...")
            img=getFrame(*args)
            #print ("Saving array to tfile...")
            #use numpys built in save function to convert image to bytes
            np.save(Tfile,img)
            #print ("Compressing...")
            #compress it into even less bytes
            b = C.compress(Tfile.getvalue())
            #send it            
            #print("sending...")
            #lend=self.conn.sendall(b)
            send_msg(self.conn,b)
            #print("Sent {}KB".format(int(lend/1000)))

    def close(self):
        self.s.close()
    
if __name__ == "__main__":
    cam = camera.Camera()
    server = Server(port=5000)
    server.serve()
    server.startStream(lambda:cam.image)