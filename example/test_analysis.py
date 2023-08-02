import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import seaborn 
import pickle
import test_functions as tf
import sklearn.metrics as sk

data = []
tap_file = os.getcwd()+"/results/tapexplore.csv"
pickle_dir = os.getcwd()+"/data/"
# Load names from pickle file
name_file = open(pickle_dir+"pattern_names.pkl","rb")
all_names = pickle.load(name_file)
name_file.close()
# Load coordinates
mds_pos_file = open(pickle_dir+"mds_pos.pkl", 'rb')
mds_pos = pickle.load(mds_pos_file) 
mds_pos_file.close()

sections=[[]for x in range(16)]
names_in_section=[[]for x in range(16)] 
pattern_idxs = [[]for x in range(16)]
tf.make_pos_grid(mds_pos,sections,pattern_idxs,all_names,names_in_section, False) # bool is show plot

test_patterns = [894, 423, 1367, 249, 
                 939, 427, 590, 143,
                 912, 678, 1355, 580,
                 1043, 673, 1359, 736]

# Load and read CSV
with open(tap_file) as results: 
    reader = csv.reader(results)
    i=0
    for row in reader:
        data.append(row)
    results.close()
#print(len(data))
#print(len(data[0]))

# 20-35 c1
# 36-51 d1
# 52-67 c2
# 68-84 d1


sc1 = [0.0 for x in range(16)]
sc2 = [0.0 for x in range(16)]
_idx = [int(x) for x in range(16)]
idx = np.array(_idx)
idx = idx + 1
data_clean=[]
# Calculate 5th 6th alg
for test in range(len(data)):
    if test==0:
        test+=1
    for i in range(16):
        if data[test][20+i]!="" or data[test][36+i]!=0 or data[test][52+i]!=0 or data[test][68+i]!="":
            sc1[i]= float(data[test][20+i])*float(data[test][36+i])
            sc2[i]= float(data[test][52+i])*float(data[test][68+i])
            data[test].append(sc1[i])

    for i in range(16):
        data[test].append(sc2[i])
    #print(len(data[test]))

# Plot heatmap by pattern
tap_590 = []

avg = [0.0 for x in range(16)]

# change here
patt = 19
patt_num=590
#

c_one = data[patt][20:36]
d_one = data[patt][36:52]
c_two = data[patt][52:68]
d_two = data[patt][68:84]
sc_one= data[patt][-32:-16]
sc_two= data[patt][-16:]
avg=np.array(avg)
for test in range(len(data)-1):
    test += 1 
    row = data[test][4:20]
    if int(data[test][2]) ==patt_num:
        #print(test)
        tap_590.append(row)
tap_590=np.array(tap_590, dtype=float)
avg = np.mean(tap_590, axis=0)
c_two = np.array(c_two, dtype=float)
c_one = np.array(c_one, dtype=float)
sc_two = np.array(sc_two, dtype=float)
sc_one = np.array(sc_one, dtype=float)
#avg_norm = ((avg - avg.min()) / (avg.max() - avg.min()))
_avg=[0.0 for x in range(len(avg))]

# Take into account max tap strength 
#c_two = np.where(c_two < avg.max(), c_two,avg.max())

# Remove mistaps (by mean)
replace = avg
_avg = np.where(avg > np.mean(avg), avg, replace)
#print(avg)
#print(c_two)
#print(tap_590)
# Plot the array
idx2=idx-1
#plt.plot(idx2, _avg, color='darkred', marker='o', label="Tapped Value")
seaborn.boxplot(data=tap_590, color="lightcoral")
plt.plot(idx2, c_one, color='darkred', marker='o', linestyle='--', label="Predicted Value")
plt.ylabel("MIDI Velocity Value")
plt.xlabel("Step")
plt.title("New Predicted vs Tapped Rhythm")
plt.ylim([0,1])
#plt.legend()
print(sk.mean_absolute_error(_avg, c_two))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the first array on the first subplot

ax1.bar(idx, _avg, color='darkred')
# Add labels and title
ax1.set_xlabel('Step')
ax1.set_ylabel('Tapped Value')
ax1.set_title('Average Tapped Rhythm')
ax1.set_ylim([0, 1])

ax2.bar(idx, c_two, color='grey')
ax2.set_xlabel('Step')
ax2.set_ylabel('Predicted Value')
ax2.set_title('Predicted Rhythm with Alg=[DensSyncMeter Continuous]')
ax2.set_ylim([0, 1])

fig.tight_layout()

# Show the plot
plt.show()