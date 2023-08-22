import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import seaborn 
import pickle
import test_functions as tf
import sklearn.metrics as sk
import warnings

with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib\..*")


# LOOK HERE BEFORE RUNNING
# Select which graphs to show
_coordinates = True
_controlcomparison = True
_subjectaverageerror = True
_patternaverageerror = True
_tapcalibration = True

n_subjects=20


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
# Load calibration data
taptap = []
taptap_f = os.getcwd()+"/results/taptest.csv"

# plot coordinates and selected patterns
sections=[[]for x in range(16)]
names_in_section=[[]for x in range(16)] 
pattern_idxs = [[]for x in range(16)]
tf.make_pos_grid(mds_pos,sections,pattern_idxs,all_names,names_in_section, _coordinates) # bool is show plot
test_patterns = [894, 423, 1367, 249, 939, 427, 590, 143, 912, 678, 1355, 580, 1043, 673, 1359, 736]
test_patterns_alg = [678, 1355, 894, 423, 1367, 249, 939, 427, 590, 143, 912, 678, 1355, 580, 1043, 673, 1359, 736]

# Load and read CSV
with open(tap_file) as results: 
    reader = csv.reader(results)
    for row in reader:
        data.append(row)
    results.close()
with open(taptap_f) as results:
    reader = csv.reader(results)
    for row in reader:
        taptap.append(row)
    results.close()

# Add in last algorithm
sc1 = [0.0 for x in range(16)]
sc2 = [0.0 for x in range(16)]
_idx = [int(x) for x in range(16)]
idx = np.array(_idx, dtype=int)
idx = idx + 1
data_anova=[[0.0 for x in range(len(data)-1)] for y in range(6)]
by_person =[[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects)] # [patt# tapresults]x18tests
by_alg = [[[0.0 for x in range(17)] for y in range(18)] for z in range(6)] # algs



# Sort through data
counts = np.array([0 for x in range(16)], dtype=float)
avgs = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)
algs = np.array([[[0.0 for x in range(16)] for y in range(6)]for z in range(16)], dtype=float)
alg_names = ["cont1","disc1","cont2","disc2","semicont1","semicont2"]
t_count=0
p_count=0
for i in range(len(data)):
    if i!=0: # skip first row (is header)
        md=[0.0 for b in range(6)] #mean distance
        md=np.asarray(md, dtype=float)
        tap = np.asarray(data[i][4:20], dtype=float)
        c_one = np.asarray(data[i][20:36], dtype=float)
        d_one = np.asarray(data[i][36:52], dtype=float)
        c_two = np.asarray(data[i][52:68], dtype=float)
        d_two = np.asarray(data[i][68:84], dtype=float)
        sc_one= np.asarray(data[i][-32:-16], dtype=float)
        sc_two= np.asarray(data[i][-16:], dtype=float)
        for j in range(16):
            md[0] += tap[j]-c_one[j]
            md[1] += tap[j]-c_two[j]
            md[2] += tap[j]-d_one[j]
            md[3] += tap[j]-d_two[j]
            md[4] += tap[j]-sc_one[j]
            md[5] += tap[j]-sc_two[j]
        md = md / 16   
        for j in range(6):
            data_anova[j][i-1] = md[j]     

        # add tap to avg for pattern, count how many times pattern shows up
        for y in range(len(test_patterns)):
            if int(data[i][2]) == test_patterns[y]:
                #print(f"{y}-{test_patterns[y]}")
                avgs[y] += tap
                counts[y] += 1
                algs[y][0] = c_one
                algs[y][1] = d_one
                algs[y][2] = c_two
                algs[y][3] = d_two
                algs[y][4] = sc_one
                algs[y][5] = sc_two
        for y in range(len(test_patterns_alg)):
            if int(data[i][2])==test_patterns_alg[y]:
                by_alg[0][y][1:] = c_one
                by_alg[1][y][1:] = d_one
                by_alg[2][y][1:] = c_two
                by_alg[3][y][1:] = d_two
                by_alg[4][y][1:] = sc_one
                by_alg[5][y][1:] = sc_two
                by_alg[0][y][0] = test_patterns_alg[y]
                by_alg[1][y][0] = test_patterns_alg[y]
                by_alg[2][y][0] = test_patterns_alg[y]
                by_alg[3][y][0] = test_patterns_alg[y]
                by_alg[4][y][0] = test_patterns_alg[y]
                by_alg[5][y][0] = test_patterns_alg[y]

        # find by person
        line = [0.0 for x in range(17)]
        line[0]=int(data[i][2])
        for k in range(len(tap)):
            line[k+1]=tap[k]
        if (i-1)%18==0 and (i-1)!=0:
            p_count += 1
            t_count = 0
        if(p_count!=n_subjects):
            by_person[p_count][t_count] = np.asarray(line, dtype=float)
            t_count +=1
avgs = avgs / counts


# Remove outliers here (only for graphin)
_removeoutliers = True
if _removeoutliers:
    n_outliers = 1
    by_person2 =[[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects-n_outliers)] # [patt# tapresults]x18tests
    outliers = [6]
    by_person2 = np.delete(by_person, outliers, axis=0)
    by_person=by_person2
    print(len(by_person))
    n_subjects = n_subjects-n_outliers

# Find and compare single pattern and alg, with error bars
by_pattern = [[] for x in range(16)]
_by_pattern = True # bool for plotting individual patts
for i in range(len(data)):
    if i!=0: # skip first row
        for j in range(len(test_patterns)):
            if int(data[i][2]) == test_patterns[j]: # if is pattern #, get tap
                by_pattern[j].append(np.asarray(data[i][4:20], dtype=float))

patt_means = [[0.0 for x in range(16)] for x in range(16)]
patt_stds = [[0.0 for x in range(16)] for x in range(16)]
patt_vars = [[0.0 for x in range(16)] for x in range(16)]
for patt in range(len(by_pattern)):
    patt_means[patt] = np.mean(by_pattern[patt], axis=0)
    patt_stds[patt] = np.std(by_pattern[patt], axis=0)
    patt_vars[patt] = np.var(by_pattern[patt], axis=0)


# by_person[#][test][patt#-results] - [n_subjects][18][17]
# by_pattern[patt][#_tests]


# Necessary to change, otherwise many fix
by_all = [[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects+6)]
by_all[:n_subjects] = by_person
by_all[-6:] = by_alg
by_person=by_all


## Compare subjects summed avg MAE results from control pattern 1 & control pattern 2
# Control 1 = 678, test_patterns[9]
# Control 2 = 1355, test_patterns[10]
if _controlcomparison:
    control1_avg = patt_means[9]
    control2_avg = patt_means[10]
    control_errors = np.array([[0.0,0.0] for x in range(n_subjects+6)], dtype=float)

    for person in range(len(by_person)):
        #print(person)
        for test in range(len(by_person[person])):
            if by_person[person][test][0]==678:
                control_errors[person][0] += np.mean(np.abs(by_person[person][test][1:]-control1_avg))
            if by_person[person][test][0]==1355:
                control_errors[person][1] += np.mean(np.abs(by_person[person][test][1:]-control2_avg))
            print(f"{control_errors[person]}-{person}")

    fig, (ax, ax1) = plt.subplots(1,2,figsize=(12,6))

    bp = ax.boxplot(control_errors, patch_artist=True, boxprops=dict(facecolor='none'))
    ax.scatter(np.full(len(control_errors)-6, 1), control_errors[:n_subjects,0],color='lightcoral', linewidth=0.75, marker='x', label="Sum Err. 678")
    ax.scatter(np.full(len(control_errors)-6, 2), control_errors[:n_subjects,1],color='lightcoral', linewidth=0.75, marker='x', label="Sum Err. 1355")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 1), control_errors[n_subjects:,0],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 678")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 2), control_errors[n_subjects:,1],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 1355")

    ax.set(xticks=[1,2], xticklabels=[str(x) for x in [678,1355]], ylim=[-0.1, 1], xlim=[0,3])
    ax.set_title(f"Summed MAE from Control Patterns", fontsize=14, fontfamily='serif')
    ax.set_xlabel("Control Test Pattern", fontsize=12, fontfamily='serif')
    ax.set_ylabel("Summed MAE from Mean Tapped Pattern", fontsize=12, fontfamily='serif')
    ax.axhline(y=0,color='black', alpha=0.6, linestyle='--')

    ax1.scatter(control_errors[:n_subjects,0], control_errors[:n_subjects,1], marker='x', color='lightcoral')
    ax1.scatter(control_errors[n_subjects:,0], control_errors[n_subjects:,1], marker='x', color='teal', linewidth=1)

    ax1.set(xlim=[0,1], ylim=[0,1])
    for n in range(n_subjects):
        ax1.text(control_errors[n,0], control_errors[n,1], str(n), size='x-small' )
    lbls=["[C1]","[D1]","[C2]","[D2]","[SC1]","[SC2]"]
    for n in range(6):
        ax1.text(control_errors[n_subjects+n,0], control_errors[n_subjects+n,1], lbls[n], size='x-small')
    ax1.set_title("Subject Control Pattern Cross Errors", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Summed MAE for Pattern 678", fontfamily='serif')
    ax1.set_ylabel("Summed MAE for Pattern 1355", fontfamily= 'serif')
    ax1.grid(color='lightgrey', linewidth=1, alpha=0.4)
    plt.show()



if _subjectaverageerror:
    # Subject average error (over all patterns) x subject
    control_patterns = np.array([[[0.0 for x in range(16)] for y in range(4)] for z in range(n_subjects+6)],dtype=float)
    control_differences = np.array([[0.0 for y in range(2)] for z in range(n_subjects+6)], dtype=float)
    sae = np.array([0.0 for x in range(n_subjects+6)])
    sae_box = np.array([[0.0 for y in range(18)] for z in range(n_subjects+6)])
    control1=678
    c1_idx = 0
    control2=1355
    c2_idx = 2

    
    ##############
    for person in range(len(by_person)):
        mean_diff=np.array([[0.0 for x in range(16)] for y in range(18)])
        stds=np.array([[0.0 for x in range(16)] for y in range(18)])

        for test in range(len(by_person[person])):
            # get answers for two control patterns
            if by_person[person][test][0] == control1:
                control_patterns[person][c1_idx]=by_person[person][test][1:]
                c1_idx+=1
            if by_person[person][test][0] == control2:
                control_patterns[person][c2_idx]=by_person[person][test][1:]
                c2_idx+=1
            
            # check which pattern it is
            for patt in range(len(test_patterns)): 
                if by_person[person][test][0]==test_patterns[patt]:
                    mean_diff[test] =  np.abs(by_person[person][test][1:] - patt_means[patt])
                    #mean_diff[test] =  by_person[person][test][1:] - patt_means[patt]
            
            sae_box[person][test] = np.mean(mean_diff[test])
            sae[person] = np.mean(sae_box[person][test])
        
        control_differences[person][0]=np.mean(np.abs([(control_patterns[person][0]-patt_means[9]),(control_patterns[person][1]-patt_means[9])]))
        control_differences[person][1]=np.mean(np.abs([(control_patterns[person][2]-patt_means[10]),(control_patterns[person][3]-patt_means[10])]))
        c1_idx=0
        c2_idx=2
        
        
    # Plot of MAE  of subjects
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    ax.axhline(y=0,color='black', alpha=0.8, linewidth=1,label='No Error')
    s_idx = np.array(np.arange(n_subjects+6))
    labels = s_idx+1
    lables = [str(x) for x in labels]
    lables[-6:] = ["[C1]","[D1]","[C2]","[D2]","[SC1]","[SC2]"]
    for i in range(len(by_person)):
        ax.scatter(np.full(len(sae_box[i]),i+1),sae_box[i], color='lightcoral', linewidth=0.5, marker='x', alpha=0.6)
    ax.scatter(s_idx+1,control_differences[:,0],color="purple",marker='x', label='Subject Control Error (678)',linewidth=1)
    ax.scatter(s_idx+1,control_differences[:,1],color="teal",marker='x', label='Subject Control Error (1355)',linewidth=1)
   
    custom_colors = ['none'] * n_subjects + ['lightgreen'] * 6
    bp=ax.boxplot(sae_box.T, patch_artist=True)
    for box, color in zip(bp['boxes'], custom_colors):
        box.set_facecolor(color)
        if color == "lightgreen":
            box.set_alpha(0.2)
    ax.scatter(s_idx+1, sae, color='coral', linestyle='-', marker='x', label="Subject Avg. Err.")
    ax.set_title(f"Subject v. Subject Average Error (all patts)", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Subject #", fontsize=13, fontfamily='serif')
    ax.set_ylabel("Average Velocity Error from \n All Mean Tapped Patterns", fontsize=13, fontfamily='serif')
    ax.set(xticks=s_idx+1, xticklabels=[str(x) for x in lables], ylim=(-0.1,1))

    plt.legend()
    plt.show()

if _patternaverageerror:
    pae = np.array([0.0 for x in range(len(test_patterns))])
    pae_box = np.array([[0.0 for y in range(len(by_person))] for z in range(len(test_patterns))])
    for person in range(len(by_person)): # per subject:
        mean_diff=np.array([[0.0 for x in range(16)] for y in range(18)])
        stds=np.array([[0.0 for x in range(16)] for y in range(18)])
        #[894, 423, 1367, 249, 939, 427, 590, 143, 912, 678, 1355, 580, 1043, 673, 1359, 736]
        # go through subjects tests
        for test in range(len(by_person[person])):
            for patt in range(len(test_patterns)): # check which pattern it is
                if by_person[person][test][0]==test_patterns[patt]:
                    mean_diff[test] =  np.abs(by_person[person][test][1:] - patt_means[patt])
                    #mean_diff[test] =  by_person[person][test][1:] - patt_means[patt]
                
                    pae_box[patt][person] = np.mean(mean_diff[test])
    
    for patt in range(len(test_patterns)):
        pae[patt]=np.mean(pae_box[patt])
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    pidx=np.arange(16)+1

    ax.axhline(y=0,color='black', alpha=0.6)
    for i in range(len(test_patterns)):
        ax.scatter(np.full(len(pae_box[i]),i+1),pae_box[i], color='lightcoral', linewidth=0.5, marker='x', alpha=0.6)
    bp= ax.boxplot(pae_box.T, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    ax.scatter(pidx, pae,color='lightcoral', linestyle='-', marker='x', label="Pattern Avg. Err.")
    
    ax.set(xticks=pidx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.1,1))
    ax.set_title(f"Mean Velocity Error from Mean by Pattern", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Test Pattern", fontsize=12)
    ax.set_ylabel("Mean Velocity Error (tap velocity)", fontsize=12, fontfamily='serif')
    plt.show()



## TODO:
# - Make function to remove subject 6, and maybe 9.
# - Remove second occasion of the control pattern (each only will have 16 trials)
# - Use MAE for quant stuff. Average error can be used for subjective analysis (hard or soft tappers etc)
# - [DONE] Bring in algorithms for comparison
#








for line in range(len(taptap)):
    for tap in range(32):
        taptap[line][tap+3]==float(taptap[line][tap+3])

## Tap Calibration
if _tapcalibration:
    print(len(taptap))
    size = int(len(taptap)/3)
    taps = [[[0.0 for x in range(32)] for y in range(3)] for z in range(size)]
    taps_lmh = [[[0.0 for x in range(32)] for y in range(size)] for z in range(3)]
    taps_mid = [[0.0 for x in range(32)] for x in range(size)]
    taps_high = [[0.0 for x in range(32)] for x in range(size)]
    taps_low = [[0.0 for x in range(32)] for x in range(size)]
    count=0
    mean_high=np.array([0.0 for x in range(32)], dtype=float)
    mean_mid=np.array([0.0 for x in range(32)], dtype=float)
    mean_low=np.array([0.0 for x in range(32)], dtype=float)

    # by participant
    for line in range(len(taptap)):
        test = int(taptap[line][2])
        #print(f"line {line} count {count} test {test} {taptap[line][2]}")
        taps_lmh[test-1][count] = taptap[line][3:]
        if test==1:
            taps[count][0]=taptap[line][3:]
            taps_low[count]=taptap[line][3:]
            mean_low += np.array(taptap[line][3:], dtype=float)
        elif test==2:
            taps[count][1]=taptap[line][3:]
            taps_high[count]=taptap[line][3:]
            mean_high += np.array(taptap[line][3:], dtype=float)
        elif test==3:
            taps[count][2]=taptap[line][3:]
            taps_mid[count]=taptap[line][3:]
            mean_mid += np.array(taptap[line][3:], dtype=float)
        if line%3==0 and line!=0:
            count+=1
    mean_high /= size
    mean_mid /=size
    mean_low /=size
    tapidx=np.arange(32)
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    for k in range(int(len(taptap)/3)):
        ax.scatter(tapidx+1,np.array(taps_low[k], dtype=float), color='lightcoral', marker='x', linestyle='--', linewidth=1)
        ax.scatter(tapidx+1,np.array(taps_mid[k], dtype=float), color='lightgreen', marker='x', linestyle='--', linewidth=1)
        ax.scatter(tapidx+1,np.array(taps_high[k], dtype=float), color='lightblue', marker='x', linestyle='--', linewidth=1)

    ax.plot(tapidx+1, mean_high, color='blue', label='Mean Tap High')
    b=ax.boxplot(np.array(taps_high,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, color='lightblue'))
    ax.plot(tapidx+1, mean_mid, color='green', label='Mean Tap Mid')
    b2=ax.boxplot(np.array(taps_mid,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, color='lightgreen'))
    b3=ax.boxplot(np.array(taps_low,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, color='lightcoral'))
    ax.plot(tapidx+1, mean_low, color='red', label='Mean Tap Low')
    ax.axhline(y=42,color='black', alpha=0.6, linestyle='--')
    ax.axhline(y=84,color='black', alpha=0.6, linestyle='--')
    #ax.hist(np.array(taps_high,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, color='lightblue'))

    ax.set(ylim=[0,127], yticks=[0,42,84,127], xticks=tapidx+1)
    ax.set_title(f"Progressive Tapping Consistency over all Subjects")
    ax.set_xlabel("Tap Order")
    ax.set_ylabel("Tapped Velocity")
    plt.legend(loc='best')
    plt.show()