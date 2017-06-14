import socket; 
print socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("localhost", 49912))
