import socket
import pickle
import io

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('18.111.87.85', 444))

def recv(size=1024):
    data = bytearray()
    while 1:
        buffer = s.recv(1024)
        if len(buffer)>0:
            data+=buffer
        else:
            return data

while 1:
    b = io.BytesIO(recv())
    img = pickle.load(b)
    cv2.imshow(img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit

