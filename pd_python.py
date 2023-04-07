import socket
import time
import struct

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

# UDP Definition
UDP_IP = "127.0.0.1"
UDP_PORT = 1337
sockt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # Internet and UDP
print("Connecting to"+UDP_IP+":"+str(UDP_PORT))

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
'''pattlist=[
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
'''
pd_arr = [0]*(len(pattlist))
s = ''
pd = []
for i in range(len(pattlist)):
    s+=str(i)
    for event in enumerate(pattlist[i]):
        s+=' '
        #print(event[1])
        s = s + str(event[1])
        s2=str(i)+' '+str(event[1])
        pd.append(s2)
        s2 = ''
    pd_arr[i]=s
    s = ''
#print(pd_arr)
#print(pd)

t = 0.01

for j in pd:
    sockt.sendto(j, (UDP_IP, UDP_PORT))
    print("sent: "+str(j))
    time.sleep(t)

sockt.close()