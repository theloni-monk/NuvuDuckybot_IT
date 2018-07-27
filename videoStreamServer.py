import socket
import cv2
import camera
import io
import numpy as np
from tempfile import TemporaryFile
import zstandard

class Server:
    def __init__(self,**kwargs):
        s = socket.socket()
        s.bind(('', kwargs.get("port",444)))
        s.listen(10)
        self.s = s
        self.verbose = kwargs.get("verbose",True)
    def serve(self):
        while True:
            self.conn, self.clientAddr = self.s.accept()
            if self.verbose:
                print('Connected with ' + self.clientAddr[0] + ':' + str(self.clientAddr[1]))
    def startStream(self,getFrame,args=[]):
        #not sure if np.save compression is worth io overhead...
        Tfile=TemporaryFile()
        C=zstandard.ZstdCompressor()
        while True:
            #fetch the image
            img=getFrame(*args)
            #use numpys built in save function to convert image to bytes
            np.save(Tfile,img)
            #compress it into even less bytes
            b = io.BytesIO(C.compress(Tfile.read(Tfile.tell()).encode()))
            #send it            
            self.conn.send(b.getvalue())
            print("Sent {}KB".format(int(len(r)/1000)))
    def close(self):
        self.s.close()
    
if __name__ == "__main__":
    cam = camera.Camera()
    server = Server()
    server.serve()
    server.startStream(lambda:cam.image)