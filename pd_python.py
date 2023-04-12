import socket
import time
import struct
import json
import os

# Change for respective usage; must be the monophonic-flattenning folder
os.chdir("/Users/peterclark/Documents/Barcelona/22-23/Thesis/PureData/MonophonicFlattening/")

# Structures
'''
MONO PATTLIST EXAMPLE
pattlist=[
	[1, 2, 3],
	[2],
	[3],
	[],
	[1, 2, 3],
	[2],
	[3],
	[],
	[1, 2, 3],
	[2],
	[3],
	[],
	[1, 2, 3],
	[2],
	[3],
	[1,2,3]
    ]

POLY PATTLIST EXAMPLE
pattlist=[
	[36, 42],
	[42],
	[42, 46, 45],
	[42],
	[36, 38, 42],
	[42, 46],
	[],
	[42],
	[36, 42],
	[42],
	[42, 46, 45],
	[42],
	[36, 38, 42],
	[42, 46],
	[42],
	[42]
    ]
'''

GM_dict={
# key is midi note number
# values are:
# [0] name (as string)
# [1] name category low mid or high (as string)
# [2] substiture midi number for simplified MIDI (all instruments)
# [3] name of instrument for 8 note conversion (as string)
# [4] number of instrument for 8 note conversion
# [5] substiture midi number for conversion to 8 note
# [6] substiture midi number for conversion to 16 note
# [7] substiture midi number for conversion to 3 note
# if we are going to remap just use GM_dict[msg.note][X]
    22:['Closed Hi-Hat edge', 'high', 42, 'CH', 3,42,42,42],
    26:['Open Hi-Hat edge', 'high', 46, 'OH', 4,46,46,42],
    35:['Acoustic Bass Drum','low',36, 'K', 1, 36,36,36],
    36:['Bass Drum 1','low',36, 'K', 1, 36,36,36],
    37:['Side Stick','mid',37, 'RS', 6, 37,37,38],
    38:['Acoustic Snare','mid',38, 'SN', 2, 38,38,38],
    39:['Hand Clap','mid',39, 'CP', 5, 39, 39,38],
    40:['Electric Snare','mid',38, 'SN', 2, 38,38,38],
    41:['Low Floor Tom','low',45, 'LT', 7, 45,45,36],
    42:['Closed Hi Hat','high',42, 'CH', 3, 42,42,42],
    43:['High Floor Tom','mid',45, 'HT', 8, 45,45,38],
    44:['Pedal Hi-Hat','high',46, 'OH', 4, 46, 46,42],
    45:['Low Tom','low',45, 'LT', 7, 45, 45,36],
    46:['Open Hi-Hat','high',46, 'OH', 4, 46, 46,42],
    47:['Low-Mid Tom','low',47, 'MT', 7, 45, 47,36],
    48:['Hi-Mid Tom','mid',47, 'MT', 7, 50, 50,38],
    49:['Crash Cymbal 1','high',49, 'CC', 4, 46, 42,42],
    50:['High Tom','mid',50, 'HT', 8, 50, 50,38],
    51:['Ride Cymbal 1','high',51, 'RC', -1, 42, 51,42],
    52:['Chinese Cymbal','high',52, '', -1, 46, 51,42],
    53:['Ride Bell','high',53, '', -1, 42, 51,42],
    54:['Tambourine','high',54, '', -1, 42, 69,42],
    55:['Splash Cymbal','high',55, 'OH', 4, 46, 42,42],
    56:['Cowbell','high',56, 'CB', -1, 37, 56,42],
    57:['Crash Cymbal 2','high',57,'CC', 4,46, 42,42],
    58:['Vibraslap',"mid",58,'VS', 6,37, 37,42],
    59:['Ride Cymbal 2','high',59, 'RC',3, 42, 51,42],
    60:['Hi Bongo','high',60, 'LB', 8, 45,63,42],
    61:['Low Bongo','mid',61, 'HB', 7, 45, 64,38],
    62:['Mute Hi Conga','mid',62, 'MC', 8, 50, 62,38],
    63:['Open Hi Conga','high',63, 'HC', 8, 50, 63,42],
    64:['Low Conga','low',64, 'LC', 7, 45,64,36],
    65:['High Timbale','mid',65, '',8, 45,63,38],
    66:['Low Timbale','low',66, '',7, 45,64,36],
    67:['High Agogo','high',67, '',-1, 37,56,42],
    68:['Low Agogo','mid',68,'',- 1 , 37,56,38],
    69:['Cabasa','high',69, 'MA',-1, 42,69,42],
    70:['Maracas','high',69, 'MA',-1, 42,69,42],
    71:['Short Whistle','high',71,'',-1,37, 56,42],
    72:['Long Whistle','high',72,'',-1,37, 56,42],
    73:['Short Guiro','high',73,'',-1, 42,42,42],
    74:['Long Guiro','high',74,'',-1,46,46,42],
    75:['Claves','high',75,'',-1, 37,75,42],
    76:['Hi Wood Block','high',76,'',8, 50,63,42],
    77:['Low Wood Block','mid',77,'',7,45, 64,38],
    78:['Mute Cuica','high',78,'',-1, 50,62,42],
    79:['Open Cuica','high',79,'',-1, 45,63,42],
    80:['Mute Triangle','high',80,'',-1, 37,75,42],
    81:['Open Triangle','high',81,'',-1, 37,75,42],
    }

# Mapping instruments
mapped_instruments=[]
for key in GM_dict.keys():
	mapped_instruments.append(GM_dict[key][5])
set_of_instruments=list(set(mapped_instruments))

def find_8note(note):
    if(int(note)==0):
        return []
    note_8 = GM_dict[int(note)][5]
    return note_8

def find_3note(note):
    if(int(note)==0):
        return []
    channel = GM_dict[int(note)][1]
    if(channel=="low"):
        note_3 = 10 # not 1 as those are used for mono channels
    elif(channel=="mid"):
        note_3 = 11
    else:
        note_3 = 12 # "high"
    return note_3

def save_and_extract_patternlist(filename, size, save_bool):
    pattern_list = []

    #extract drum pattern txt to list
    with open(filename) as file:
        file_contents = []
        for line in file:
            file_contents.append(line)

        for idx, content in enumerate(file_contents):
            line = content[:len(content)-1]
            n = line.split(" ")[0].replace(",","") # select 'name, '
            if n=="name": # indicates there is a pattern following / name line
                pattern = []
                pattern_3 = []
                genre = line.split(" ")[1]
                pattern.append(genre)
                for i in range(size):
                    true_idx = i+1+idx # account for step #, the name line, and location in txt doc
                    step = (file_contents[true_idx][:len(file_contents[true_idx])-2].split(", ")[1])
                    s = step.split(" ") # separate into array
                    s_8 = []
                    s_3 = []
                    for i, note in enumerate(s):
                        s[i]=int(note)
                        if s[i]==0:
                            s_8 = [] # for pd formatting, empty step
                        else:
                            s_8.append(find_8note(s[i])) #convert to 8 note
                            s_3.append(find_3note(s[i]))
                            s_8[i]=int(s_8[i])
                    #print(s_8)
                    #print(s_3)
                    pattern.append(s_8)
                    pattern_3.append(s_3)
                #print(pattern)
                #print(pattern_3)
                fullpatt = monophonic_reductions(pattern_3, pattern)
            
                #save_pattern16(pattern, genre, save_bool)
                for i in range(len(fullpatt)):
                   pattern.append(fullpatt[i])
                pattern_list.append(pattern)
                print(len(pattern_list))
    return pattern_list

def extract_pattern(filename, size):
    pattern_list = []

    #extract drum pattern txt to list
    with open(filename) as file:
        file_contents = []
        for line in file:
            file_contents.append(line)
        
        f = str(file_contents[0])
        pattern_raw = []
        left_idx=0
        #print(f)

        # SEPARATE INTO NAME AND STEPS
        for c in range(len(f)-1):
            if f[c] == '"' and f[c+1] ==",":
                pattern_raw.append(f[2:c]) # add name
            if f[c]=='[' and (f[c+1].isdigit()==True or f[c+1]==']'): #if a note or empty note
                left_idx=c+1
            if f[c]==']' and (f[c+1]==',' or f[c+1]==']'):
                l = f[left_idx:c].replace(",","")
                pattern_raw.append(l)

        #for i in range(len(pattern_raw)):
        #    print(pattern_raw[i])

        # EXTRACT MIDI AND MONO INFO
        genre = pattern_raw[0]
        patt = pattern_raw[1:]
        print("genre: "+str(genre))
        pattern = []
        pattern.append(genre)
        pattern_3 = []
        #print(len(pattern_raw))
        for i in range(size):
            step = patt[i]
            s = step.split(" ")
            s_8=[]
            s_3=[]
            for j,note in enumerate(s):
                if(note!=''):
                    s[j]=int(note)
                    s_8.append(find_8note(s[j])) #convert to 8 note
                    s_3.append(find_3note(s[j]))
                    s_8[j]=int(s_8[j])
                    s_3[j]=int(s_3[j])
                else:
                    s_8=[]
                    s_3=[]
            pattern.append(s_8)
            pattern_3.append(s_3)
        fullpatt = monophonic_reductions(pattern_3,pattern)
        for i in range(len(fullpatt)):
            pattern.append(fullpatt[i])
        pattern_list.append(pattern)
    return pattern_list

def save_pattern16(pattern, fn, save_bool):
    if save_bool:
        file_name = "Patterns/"+fn+".txt"
        #if file doesnt exist
        if not os.path.exists(file_name):
            with open(file_name, "a+") as file:
                json.dump(pattern, file)

def monophonic_reductions(patt, poly_patt):
    patt_low=[0]*16
    patt_mid=[0]*16
    patt_high=[0]*16
    mono_patterns = [[0 for x in range(16)] for y in range(5)]
    mono_master = [[] for x in range(16)]
    poly_patt = poly_patt[1:]
    for i in range(len(patt)):
        for j in range(len(patt[i])):
            if(patt[i][j]==10):
                patt_low[i]+=1
            elif(patt[i][j]==11):
                patt_mid[i]+=1
            elif(patt[i][j]==12):
                patt_high[i]+=1
    
    sync_sal_profile16 = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1] #Longuet-Higgen's Lee
    memory_residual_profile16 = [1,1.05,0.75,-0.75, 1.5,-0.4,-0.5,0, 1.25,-0.5,0.25,-0.25, 0,-1.25,-1,0.25] #PalmerKrumhansl
    threshold = 0.49 #arbitrary
    sync_threshold = 4.5 #arbys
    weights = [0.5, 0.3, 0.1] #arbitrary

    for i in range(len(patt)):
        if(patt_low[i]>0 or patt_mid[i]>0 or patt_high[i]>0): # there exists a note on the beat
            
            # Naive reduction
            mono_patterns[0][i] = 1
            mono_master[i].append(1)

            # Weighted multi-hit channels w/ threshold of 1 
            mono_patterns[1][i] = 2 if((patt_low[i]*weights[0] + patt_mid[i]*weights[1] + patt_high[i]*weights[2])>threshold) else 0
            if mono_patterns[1][i] == 2:
                mono_master[i].append(2)

            # Syncopated Salience Profile
            pl = 1 if patt_low[i]>0 else 0
            pm = 1 if patt_mid[i]>0 else 0
            ph = 1 if patt_high[i]>0 else 0
            mono_patterns[2][i] = 3 if(((pl*sync_sal_profile16[i])+(pm*sync_sal_profile16[i])+(ph*sync_sal_profile16[i]))>sync_threshold) else 0
            if mono_patterns[2][i] == 3:
                mono_master[i].append(3)


        # Presence of Kick, Snare, or Clap (sounds that seem rhythmically reinforcing)    
        for note in poly_patt[i]:
            if((note==36) or (note==38) or (note==39)):
                mono_patterns[3][i] = 4
                mono_master[i].append(4)

    # mono patterns = array with len 16 containing monochannels
    return mono_master

# UDP Definition
UDP_IP = "127.0.0.1"
UDP_PORT = 1337
sockt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet and UDP
print("")
print("------------------------------")
print("Connecting to "+UDP_IP+":"+str(UDP_PORT))
print("")

def to_PureData(pattern_list):
    patt = pattern_list
    print(patt)
    pd=[]
    for i in range(len(pattern_list)):
        patt = pattern_list[i]
    print(patt)
    name = patt[0]
    pd = patt[1:]
    #print(pd)
    t=0.01
    for j, note in enumerate(pd):
        for i in note:
            j_16 = j%16
            idx = str(j_16)
            n = str(i)
            data = (idx+" "+n+" "+str(name))
            sockt.sendto(str(data), (UDP_IP, UDP_PORT))
            #print("sent: "+str(data))
            time.sleep(t)
    print("Pattern sent: "+str(name))

    return pd

# ---------------------------------------------------------------------- #
# Main Function

dir = os.getcwd()+"/Patterns/"
dir_list = os.listdir(dir)
for item in dir_list:
    print(item)
filename = raw_input("Type the pattern you want to explore:")
if filename.endswith(".txt")==0:
    filename = filename+".txt"
filename = dir+filename
size=16
save_bool=False
#pattern_list = save_and_extract_patternlist(filename, size, save_bool)
pattern_list = extract_pattern(filename, size)
pd = to_PureData(pattern_list)
print("------------------------------")
print("")

sockt.close()
# ---------------------------------------------------------------------- #
