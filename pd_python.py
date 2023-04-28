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
        #if int(note)==45:
        #    print(note)
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

        # SEPARATE INTO NAME AND STEPS
        for c in range(len(f)-1):
            if f[c] == '"' and f[c+1] ==",":
                pattern_raw.append(f[2:c]) # add name
            if f[c]=='[' and (f[c+1].isdigit()==True or f[c+1]==']'): #if a note or empty note
                left_idx=c+1
            if f[c]==']' and (f[c+1]==',' or f[c+1]==']'):
                l = f[left_idx:c].replace(",","")
                pattern_raw.append(l)

        # EXTRACT MIDI AND MONO INFO
        genre = pattern_raw[0]
        patt = pattern_raw[1:]
        print("genre: "+str(genre))
        pattern = []
        pattern.append(genre)
        pattern_3 = []
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
        fullpatt, dass_velocity = monophonic_reductions(pattern_3,pattern)
        for i in range(len(fullpatt)):
            pattern.append(fullpatt[i])
        pattern_list.append(pattern)
    return pattern_list, dass_velocity

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
    mono_patterns = [[0 for x in range(16)] for y in range(10)]
    mono_master = [[] for x in range(16)]
    poly_patt = poly_patt[1:] # Clip name

    total_low=0
    total_mid=0
    total_high=0
    total_patt=0


    # From find_3number() to counting #number of hits at timestep
    # Also counts total # of hits in overall channel and pattern 
    for i in range(len(patt)):
        for j in range(len(patt[i])):
            if(patt[i][j]==10):
                patt_low[i]+=1
                total_low+=1
                total_patt+=1
            elif(patt[i][j]==11):
                patt_mid[i]+=1
                total_mid+=1
                total_patt+=1
            elif(patt[i][j]==12):
                patt_high[i]+=1
                total_high+=1
                total_patt+=1
    
    total_low=float(total_low)
    total_mid=float(total_mid)
    total_high=float(total_high)
    total_patt=float(total_patt)
    
    sync_sal_profile16 = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1] #Original from Longuet-Higgen's Lee
    true_sync_salience = [7,1,2,1, 3,1,2,1, 4,1,2,1, 5,1,2,1] # from tests in Palmer Krumhansl
    sync_strength = [0,1,0,2, 0,1,0,3, 0,1,0,4, 0,1,0,6] # shift of -1 from P&K sync salience (and -1 to the values)

    # 1 / density%  ---> (this is # of notes in channel / total notes that appear in pattern)
    d_low = total_low/total_patt
    d_mid = float(total_mid/total_patt)
    d_high = float(total_high/total_patt)
    # salience of individual notes in channel (step agnostic)
    sal_low = 1/d_low
    sal_mid = 1/d_mid
    sal_high = 1/d_high
    sum_sal = sal_low+sal_mid+sal_high
    # normalized salience
    norm_low = sal_low/sum_sal
    norm_mid = sal_mid/sum_sal
    norm_high = sal_high/sum_sal
    
    # Arbitrary Variables for Flattening
    threshold = 0.5 #arbitrary
    sync_threshold = 4 
    sync_threshold_PK = 2
    weights = [0.5, 0.3, 0.2] #arbitrary
    mean = 0

    for i in range(len(patt)):
        if(patt_low[i]>0 or patt_mid[i]>0 or patt_high[i]>0): # there exists a note on the beat
            
            # [1] Naive reduction
            mono_patterns[0][i] = 1
            mono_master[i].append(1)

            # [2] Weighted multi-hit channels w/ threshold of 0.5 
            mono_patterns[1][i] = 2 if((patt_low[i]*weights[0] + patt_mid[i]*weights[1] + patt_high[i]*weights[2])>=threshold) else 0
            if mono_patterns[1][i] == 2:
                mono_master[i].append(2)

            # [3] Low channel only
            mono_patterns[2][i] = 3 if(patt_low[i]>0) else 0
            if mono_patterns[2][i] == 3:
                mono_master[i].append(3)

            # [5] Low channel or (mid+high)
            mono_patterns[4][i] =  5 if(patt_low[i]>0 or (patt_mid[i]>0 and patt_high[i]>0)) else 0
            if mono_patterns[4][i] == 5:
                mono_master[i].append(5)

            # [6] Syncopated Salience Profile (Longuet-Higgens & Lee) (no multihit)
            pl = 1 if patt_low[i]>0 else 0
            pm = 1 if patt_mid[i]>0 else 0
            ph = 1 if patt_high[i]>0 else 0
            mono_patterns[5][i] = 6 if(((pl*sync_sal_profile16[i])+(pm*sync_sal_profile16[i])+(ph*sync_sal_profile16[i]))>=sync_threshold) else 0
            if mono_patterns[5][i] == 6:
                mono_master[i].append(6)
            
            # [7] Reported Salience Profile (Palmer & Krumhansl) (4/4 common rhythms)
            mono_patterns[6][i] = 7 if(((patt_low[i]*true_sync_salience[i])+(patt_mid[i]*true_sync_salience[i])+(patt_high[i]*true_sync_salience[i]))>=sync_threshold_PK)else 0
            if mono_patterns[6][i] == 7:
                mono_master[i].append(7)
            
            # [8] Low, Mid without Low, High without Mid and Low ("Dominant Freq?")
            mono_patterns[7][i] = 8 if(((patt_low[i]>0) or (patt_mid[i]>0 and patt_low[i]==0) or (patt_high>0 and patt_mid[i]==0 and patt_low[i]==0)))else 0
            if mono_patterns[7][i] == 8:
                mono_master[i].append(8)

            # [9] Density & Sync Salience (DASS) Continous Output
            low_val = norm_low*patt_low[i]
            mid_val = norm_mid*patt_mid[i]
            high_val = norm_high*patt_high[i]
            if(i < len(patt)-1): # length - 1 as we use a idx+1
                if(true_sync_salience[i]<true_sync_salience[i+1]): #note is in sync pos, and is syncopated
                    if(patt_low[i+1]==0): 
                        low_val += (low_val*sync_strength[i])
                    if(patt_mid[i+1]==0): 
                        mid_val += (mid_val*sync_strength[i])
                    if(patt_high[i+1]==0): 
                        high_val += (high_val*sync_strength[i])
            mono_patterns[8][i] = (low_val+mid_val+high_val)
            mean += mono_patterns[8][i]
            if(mono_patterns[8][i]>=0.5):
                mono_master[i].append(9)

        # [4] Kick, Snare, or Clap (sounds that are rhythmically reinforcing)    
        for note in poly_patt[i]:
            if((note==36) or (note==38) or (note==39)):
                mono_patterns[3][i] = 4
                mono_master[i].append(4)
                # This can produce duplicates in the mono master, but don't seem to causing any effect on pd

    print(["{:0.2f}".format(x) for x in mono_patterns[8]])
    mean /= 16
    print(mean)
    return mono_master, mono_patterns[8]

# UDP Definition
UDP_IP = "127.0.0.1"
UDP_PORT = 1337
sockt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet and UDP
print("")
print("------------------------------")
print("Connecting to "+UDP_IP+":"+str(UDP_PORT))
print("")

def to_PureData(pattern_list, velocity_array):
    patt = pattern_list
    pd=[]
    for i in range(len(pattern_list)):
        patt = pattern_list[i]
    #print(patt)
    name = patt[0]
    pd = patt[1:]
    #print(pd)
    t=0.01
    for j, note in enumerate(pd):
        for i in note:
            j_16 = j%16
            idx = str(j_16)
            n = str(i)
            data = (idx+" "+n+" 1 "+str(name))
            sockt.sendto(str(data), (UDP_IP, UDP_PORT))
            #print("sent: "+str(data))
            time.sleep(t)
        data_vel = str(int(j%16))+" 10 "+str(velocity_array[int(j%16)])+" "+str(name)
        sockt.sendto(str(data_vel), (UDP_IP, UDP_PORT))
    print("Pattern sent: "+str(name))

    return pd

# ---------------------------------------------------------------------- #
# Main Function

dir = os.getcwd()+"/Patterns/"
dir_list = os.listdir(dir)
for item in dir_list:
    print(item)
filename = raw_input("Type the pattern you want to explore:") # py2.7 has raw_input()
if filename.endswith(".txt")==0:
    filename = filename+".txt"
filename = dir+filename
size=16
save_bool=False
#pattern_list = save_and_extract_patternlist(filename, size, save_bool)
pattern_list, velocity_dens_sync = extract_pattern(filename, size)
pd = to_PureData(pattern_list,velocity_dens_sync)
print("------------------------------")
print("")

sockt.close()
# ---------------------------------------------------------------------- #
