import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('18.111.87.85', 444))

