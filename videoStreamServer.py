import socket
import cv2
import camera

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 444))

s.listen(10)

while True:
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ':' + str(addr[1]))
    break

cam = camera.Camera()

while True:
    b = io.BytesIO()
    img = pickle.dump(cam.image,b)
    s.send(b.getvalue())
    
s.close()
