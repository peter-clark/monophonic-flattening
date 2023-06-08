import os
import sys
import numpy as np
from natsort import natsorted
import functions as fun
import descriptors as desc
import flatten as flatten
import neuralnetwork as NN
import time
import socket
import re

def parse(line):
    line = str(line)
    regex = r"[-+]?\d*\.\d+|\d" # searches for all floats or integers
    list = re.findall(regex, line)
    output = [float(x) for x in list]
    return output

# UDP Definition
UDP_IP = "127.0.0.1"
UDP_PORT = 1338
sockt = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # Internet and UDP
#print("------------------------------")
print("Connecting to "+UDP_IP+":"+str(UDP_PORT))
server_address = ('localhost', 1338)
sockt.bind(server_address)
sockt.listen(1)
tapped_pattern = [0.0 for x in range(16)] 
while True:
    print("Waiting for connection from Puredata...")
    connection, client_address = sockt.accept()
    try:
        print("client: ", client_address)
        while True:
            data = connection.recv(16)
            data = data.decode("utf-8")
            data = data.replace('\n', '').replace('\t','').replace('\r','').replace(';','')
            print(f'received {data}')
            data = data.split(" ")
            if data!='':
                tapped_pattern[int(data[1])] = float(data[0])/127
            if not data:
                break
    
    finally:
        connection.close()
        print(tapped_pattern)
