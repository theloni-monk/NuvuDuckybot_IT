from rpistream.streamclient import *
rpi = "18.111.87.85"
ian = "10.189.75.150"
clearclient = Client(serverIp=rpi, port = 5000) # Connects to the server
client.startStream() # Starts recieving data and displaying the video
