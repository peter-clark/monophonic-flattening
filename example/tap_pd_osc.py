import os
import sys
import numpy as np
from natsort import natsorted
import functions as fun
import descriptors as desc
import flatten as flatten
import neuralnetwork as NN
import to_pd as puredata
import pandas as pd
import time
import csv
import pickle
import socket
import re
import datetime
import pytz

from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_server import BlockingOSCUDPServer
from pythonosc.dispatcher import Dispatcher

###----------------------------------------------------------------###
## PATTERN INIT + SELECTION
pickle_dir = os.getcwd()+"/data/"

# Load patterns from pickle file
patt_file = open(pickle_dir+"patterns.pkl", 'rb')
all_pattlists = pickle.load(patt_file)
patt_file.close()

# Load names from pickle file
name_file = open(pickle_dir+"pattern_names.pkl","rb")
all_names = pickle.load(name_file)
name_file.close()
#all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments") # Parse all MIDI patterns in folders
pttrn = 0
pttrn = input(f"Select pattern index [0-{len(all_pattlists)}]: ")# Prompt to select pattern
patt_name = all_names[int(pttrn)]
input_patt = all_pattlists[int(pttrn)]

# Flatten pattern and organize
input_flat = flatten.flat_from_patt(input_patt)
flatterns = [[] for y in range(4)]
for i in range(len(flatterns)):
        flatterns[i] = input_flat[i]

## Empty array for tapped output to send to prediction
tappern = [0.0 for x in range(16)] # Tapped Pattern

## Empty array for saving results of testtapping
test_results = [[0 for x in range(32)] for y in range(3)] # results of three tap trials (consistency)
save_test_results = [0.0 for x in range((3+(32)))] # save line for csv (date time test# + test results)
results_file = os.getcwd()+"/results/taptest.csv"

## Empty array for results of save
save_line = [0.0 for x in range(85)]
tap_file = os.getcwd()+"/results/tapexplore.csv"

###----------------------------------------------------------------###
""" TODO:
* Define more handlers for knobs (density, etc.).
* Figure out message type for each input from PD
"""
## OSC IP / PORT
IP = '127.0.0.1'
SEND_PORT = 1338
RECEIVE_PORT = 1339
_quit=[False]
_predict=False
_save=False

## INITIALIZE MODEL FOR PREDICTION
model_dir = os.getcwd()+"/models/continuous1.pt"
model = NN.build_model()
model.load_state_dict(NN.torch.load(model_dir))

## LOAD DESCRIPTORS & EMBEDDING POSITIONS
descriptor_file = open(pickle_dir+"descriptors.pkl","rb")
d = pickle.load(descriptor_file)
descriptor_file.close()

mds_pos_file = open(pickle_dir+"mds_pos.pkl", 'rb')
mds_pos = pickle.load(mds_pos_file) 
mds_pos_file.close()

## LOAD TRIANGLES, HASH_TRIANGLES, HASH_DENSITY
triangles_file = open(pickle_dir+"triangles.pkl", 'rb')
triangles = pickle.load(triangles_file) 
triangles_file.close()

hash_density = 2 # number of col, row in space

hashed_triangles_file = open(pickle_dir+"hashed_triangles.pkl", 'rb')
hashed_triangles = pickle.load(hashed_triangles_file) 
hashed_triangles_file.close()

# Create instance of sender class
send_to_pd = SimpleUDPClient(IP, SEND_PORT)
# Create instance of receiver class
dispatcher = Dispatcher()

## Define handlers for messages
def tap_message_handler(address, *args): # /tap
    print(address)
    if address == "/tap/predict":
        global _predict
        _predict=True
        print("Predicting...")
    for idx in range(len(args)):
        tappern[idx]=(args[idx]/127) if args[idx]>=0.0 else 0.0
    print(f"Tapped Pattern: {tappern}")

def save_data_message_handler(address, *args): # save information from puredata
    # 1. DATE
    today = datetime.datetime.now(pytz.timezone("Europe/Madrid"))
    date = today.date()
    save_line[0] = date
    # 2. TIME
    time = today.strftime("%H:%M:%S")
    save_line[1] = time
    # 3. INPUT PATT
    save_line[2] = pttrn
    # 4. INPUT PATT NAME
    save_line[3] = patt_name
    # 5-20. Tapped Pattern [x16]
    for idx in range(len(tappern)):
        save_line[4+idx] = tappern[idx]
        save_line[20+idx] = flatterns[0][idx] # c1
        save_line[36+idx] = flatterns[1][idx] # d1
        save_line[52+idx] = flatterns[2][idx] # c2
        save_line[68+idx] = flatterns[3][idx] # d2
    print(f"Tapped: {save_line[4:19]}")
    print(f"CONT1 : {save_line[20:35]}")
    print(f"CONT2 : {save_line[52:67]}")
    with open(tap_file,'a') as results: 
            wr = csv.writer(results)
            wr.writerow(save_line)
            results.close()
    # SAVE TO CSV ###### 

def test_results_message_handler(address, *args): # /get_prediction (bool)
    today = datetime.datetime.now(pytz.timezone("Europe/Madrid"))
    date = today.date()
    save_test_results[0]=date
    time = today.strftime("%H:%M:%S")
    save_test_results[1]=time
    if(address=="/test_results/testnum"):
        save_test_results[2]=int(args[0])
        print(f"args {args[0]}")
    elif(save_test_results[2]>0):
        for i in range(len(args)):
            save_test_results[i+3]=args[i]
        with open(results_file,'a') as results: 
            wr = csv.writer(results)
            wr.writerow(save_test_results)
            results.close()
    if(save_test_results[2]==3):
        print("Saved Tap-Test Results to CSV.")

def joystick_message_handler(address, *args): # /joystick (xy)
     print("yadda yadda.")

def quit_message_handler(address, * args): # /quit
     _quit[0]=True
     print("I'm out.")

# Pass handlers to dispatcher
dispatcher.map("/tap*", tap_message_handler)
dispatcher.map("/save*", save_data_message_handler)
dispatcher.map("/test_results*", test_results_message_handler)
dispatcher.map("/joystick*", joystick_message_handler)
dispatcher.map("/quit*", quit_message_handler)

# Define default handler
def default_handler(address, *args):
    print(f"Nothing done for {address}: {args}")
dispatcher.set_default_handler(default_handler)

# Establish UDP connection with PureData
server = BlockingOSCUDPServer((IP, RECEIVE_PORT), dispatcher)

###----------------------------------------------------------------###
def pattern_to_pd(pattern, name, udp, type=0):
    ## Type 0 --> Input Pattern (default)
    ## Type 1 --> Predicted Pattern (midi channels *2 for PureData)
    ## Type 2 --> Flattened Patterns (different channel routing)
    if type==0:
        for step in range(len(pattern)):
            for note in range(len(pattern[step])):
                udp.send_message("/pattern/channel",pattern[step][note])
                udp.send_message("/pattern/step",step)
                udp.send_message("/pattern/velocity",1)
        print(f"Sent pattern: {patt_name}")
    if type==1:
        for step in range(len(pattern)):
            for note in range(len(pattern[step])):
                if pattern[step][note]!=0:
                    n = desc.GM_dict[int(pattern[step][note])][5]
                    udp.send_message("/pattern/channel",(n*2))
                    udp.send_message("/pattern/step",step)
                    udp.send_message("/pattern/velocity",1)
        print("Sent predicted pattern.")
    if type==2:
        #d1->1, d2->2, c1->5, c2->6, all->9
        for alg in range(4):     
            for step in range(len(pattern[0])):
                channel=9
                if alg==0: #c1
                    channel=5
                elif alg==1: #d1
                    channel=1
                elif alg==2: #c2
                    channel=6
                elif alg==3: #d2
                    channel=2
                udp.send_message("/pattern/channel",channel)
                udp.send_message("/pattern/step",step)
                udp.send_message("/pattern/velocity",pattern[alg][step])
        print("Sent flattened patterns.")
###



## PUREDATA INIT
# Parse and send selected pattern
input_patt = puredata.parse_8(input_patt) # edit this to be done in send function, not separate file
pattern_to_pd(input_patt, patt_name, send_to_pd, type=0)
pattern_to_pd(flatterns, patt_name, send_to_pd, type=2)

while _quit[0] is False:
     server.handle_request()
     if _predict:
        pred_coords = model(NN.torch.Tensor(tappern).float()).detach().numpy()
        output_patt = fun.position2pattern(pred_coords, all_pattlists,  mds_pos, triangles, hashed_triangles, hash_density)
        print(f"Predicted Pattern: {output_patt}")
        pattern_to_pd(output_patt, patt_name, send_to_pd, type=1)
        _predict=False



    # ADD A SAVE FUNCTIONALITY
    # Tapped pattern
    # Predicted coordinates
    # Input pattern
    # Name of pattern
    # Date / Time 

    ## FIX delaunay interpolation stuff wherepoint is outside any known triangles



# Example send: 
""" 
send_to_pd.send_message("/tap/velocity",velocity)
 """

