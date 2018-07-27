import socket
import cv2
import camera
import io
import pickle
import zstd

s = socket.socket()
s.bind(('', 444))

s.listen(10)

while True:
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    break

cam = camera.Camera()

while True:
    b = io.BytesIO()
    #pickle doesn't return anything, what is img
    img = pickle.dump(cam.image,b)
    print(b)
    b=zstd.compress(b)
    conn.send(b.getvalue())

s.close()
