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
s = time.perf_counter()

def parse(line):
    line = str(line)
    regex = r"[-+]?\d*\.\d+|\d" # searches for all floats or integers
    list = re.findall(regex, line)
    output = [float(x) for x in list]
    return output

def parse_8(line):
    line = str(line)
    line=line.replace("[]","[0]")
    line=line.replace("]]","")
    line=line.split("], ")
    for lin in range(len(line)):
        line[lin]=line[lin].replace("[","") 
    output = []
    for step in line:
        #print(step)
        step = step.split(", ")
        b=[]
        for note in step:
            #print(note)
            if int(note)!=0:
                b.append(get_eight_channel_note(int(note)))
        output.append(b)
    return output

def get_eight_channel_note(note):
    return desc.GM_dict[int(note)][5]


def UDP_init():
        # UDP Definition
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1337
    sockt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet and UDP
    print("------------------------------")
    print("Connecting to "+UDP_IP+":"+str(UDP_PORT))
    print("")
    return sockt

#def send_8

#--------------------------------------------------------------------------------#
_load = False
if _load:
    dir = os.getcwd()
    pattern_dir = dir+"/patterns/"
    pattern_dir_list = natsorted(os.listdir(pattern_dir))
    print(len(pattern_dir_list))
    for i in range(102,106):
        print(f"{i} - {pattern_dir_list[i]}")
    # Select pattern from directory
    pattern_index = input("Select a pattern [0-1512]:")
    pattern_filename = pattern_dir_list[int(pattern_index)]
    print(pattern_filename[:-4]) # prints åname of pattern

    # Get 8 Note Poly Representation
    eight_pattern = []
    with open(pattern_dir+pattern_filename) as file:
        file_contents = []
        for line in file:
            l = parse_8(line)
            eight_pattern=l

    # Get Flattened Patterns x 4
    flattern_dir = dir+"/flattened/"
    flattern_filename = flattern_dir+pattern_filename
    flatterns = []
    with open(flattern_filename) as file:
        file_contents=[]
        for line in file:
            l = parse(line)
            file_contents.append(l)
        for i in range(len(file_contents)):
            flatterns.append(file_contents[i])
        
    print(flatterns)
    # Send to PD channels
    sockt=UDP_init()
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1337

    for step in range(len(eight_pattern)):
        for note in range(len(eight_pattern[step])):
            data = (str(step)+" "+str(eight_pattern[step][note])+" 1 "+pattern_filename[:-4])
            sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
            time.sleep(0.01)
        data = (str(step)+" 1 "+str(flatterns[1][step])+" "+pattern_filename[:-4]) # Discrete 1
        sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
        data = (str(step)+" 2 "+str(flatterns[3][step])+" "+pattern_filename[:-4]) # Discrete 2
        sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
        data = (str(step)+" 5 "+str(flatterns[0][step])+" "+pattern_filename[:-4]) # Continuous 1
        sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
        data = (str(step)+" 6 "+str(flatterns[2][step])+" "+pattern_filename[:-4]) # Continuous 2
        sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))
        data = (str(step)+" 9 1 "+pattern_filename[:-4]) # all 16 [x]
        sockt.sendto(str(data).encode(), (UDP_IP,UDP_PORT))


    sockt.close()

#--------------------------------------------------------------------------------#
print(f"Runtime: {time.perf_counter()-s:.2f} seconds.\n")
