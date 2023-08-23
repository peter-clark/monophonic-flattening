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
_coordinates = False
_controlcomparison = False
_subjectaverageerror = True
_patternaverageerror = True
_tapcalibration = True

####
n_subjects=20
test_patterns = [894, 423, 1367, 249, 939, 427, 590, 143, 912, 580, 1043, 673, 1359, 736, 678, 1355]


## Load participant test results data.
#  For both tests.
data = []
tap_file = os.getcwd()+"/results/tapexplore.csv"
pickle_dir = os.getcwd()+"/data/"
with open(tap_file) as results: 
    reader = csv.reader(results)
    for row in reader:
        data.append(row)
    results.close()

taptap = []
taptap_f = os.getcwd()+"/results/taptest.csv"
with open(taptap_f) as results:
    reader = csv.reader(results)
    for row in reader:
        taptap.append(row)
    results.close()

# Load names and coordinates from pickle files
name_file = open(pickle_dir+"pattern_names.pkl","rb")
all_names = pickle.load(name_file)
name_file.close()
mds_pos_file = open(pickle_dir+"mds_pos.pkl", 'rb')
mds_pos = pickle.load(mds_pos_file) 
mds_pos_file.close()

# plot coordinates and selected patterns
sections=[[]for x in range(16)]
names_in_section=[[]for x in range(16)] 
pattern_idxs = [[]for x in range(16)]
tf.make_pos_grid(mds_pos,sections,pattern_idxs,all_names,names_in_section, _coordinates) 

# Calculate 5th 6th alg. Sort data into by subject for pruning
sc1 = np.array([0.0 for x in range(16)], dtype=float)
sc2 = np.array([0.0 for x in range(16)], dtype=float)
by_person =np.array([[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects)], dtype=float) # [patt# tapresults]x18tests
by_alg = np.array([[[0.0 for x in range(16)] for y in range(16)] for z in range(6)], dtype=float) # algs
control_avgs = np.array([[0.0 for x in range(16)] for y in range(2)], dtype=float)
controls = [678,1355]
p_count=0
t_count=0
print(f"{len(data)%18} & {len(data)}/18={len(data)/18}")
for test in range(len(data)):
    if test!=0:
        for i in range(16):
            sc1[i]= float(data[test][20+i])*float(data[test][36+i])
            sc2[i]= float(data[test][52+i])*float(data[test][68+i])
            data[test].append(sc1[i])
        for i in range(16):
            data[test].append(sc2[i])
        
        c_one = np.asarray(data[test][20:36], dtype=float)
        d_one = np.asarray(data[test][36:52], dtype=float)
        c_two = np.asarray(data[test][52:68], dtype=float)
        d_two = np.asarray(data[test][68:84], dtype=float)
        sc_one= np.asarray(data[test][-32:-16], dtype=float)
        sc_two= np.asarray(data[test][-16:], dtype=float)

        for y in range(len(test_patterns)):
            if int(data[test][2])==test_patterns[y]:
                by_alg[0][y] = c_one
                by_alg[1][y] = d_one
                by_alg[2][y] = c_two
                by_alg[3][y] = d_two
                by_alg[4][y] = sc_one
                by_alg[5][y] = sc_two

        # Get by-subject results
        if test!=0:
            line = [0.0 for x in range(17)]
            line[0]=int(data[test][2])
            tap = np.asarray(data[test][4:20], dtype=float)
            
            if int(data[test][2]) == controls[0]:
                control_avgs[0]+=tap
            elif int(data[test][2]) == controls[1]:
                control_avgs[1]+=tap
            
            for k in range(len(tap)):
                line[k+1]=tap[k]
            test_num = test-1
            length = len(data)-1
            if (test_num)%18==0 and (test_num)!=0:
                p_count += 1
                #print("\n")
                t_count = 0
            if(p_count!=n_subjects):
                #print(f"{test_num}/{len(data)} {p_count}-{t_count}")
                by_person[p_count][t_count] = np.asarray(line, dtype=float)
                #f line[0]==678 or line[0]==1355:
                    #print(f"{p_count} {by_person[p_count][t_count][0]} {data[test][1]} -- {line[0]}")
                t_count +=1
for y in range(2):
    control_avgs[y] = control_avgs[y] / (n_subjects*2) 

# Compare subjects summed avg MAE results from control pattern 1 & control pattern 2
# Control 1 = 678
# Control 2 = 1355
control_errors = np.array([[0.0,0.0] for x in range(n_subjects)], dtype=float)
for person in range(len(by_person)):
    for test in range(len(by_person[person])):
        if by_person[person][test][0]==678:
            control_errors[person][0] += np.mean(np.abs(by_person[person][test][1:]-control_avgs[0]))
            #print(f"{person} {int(by_person[person][test][0])}") 
        if by_person[person][test][0]==1355:
            control_errors[person][1] += np.mean(np.abs(by_person[person][test][1:]-control_avgs[1]))
            #print(f"{person} {int(by_person[person][test][0])}") 
    #print(control_errors[person])
outlier_thresholds = np.array([np.mean(control_errors[:,0])+np.std(control_errors[:,0]),np.mean(control_errors[:,1])+np.std(control_errors[:,1])], dtype=float)
print(f"Outlier Cutoff Thresholds: (678):{outlier_thresholds[0]:.4f} (1355):{outlier_thresholds[1]:.4f}")

### Plots for seeing outliers
if _controlcomparison:
    fig, (ax, ax1) = plt.subplots(1,2,figsize=(12,6))

    bp = ax.boxplot(control_errors, patch_artist=True, boxprops=dict(facecolor='none'))
    ax.scatter(np.full(len(control_errors), 1), control_errors[:,0],color='lightcoral', linewidth=0.75, marker='x', label="Sum Err. 678")
    ax.scatter(np.full(len(control_errors), 2), control_errors[:,1],color='lightcoral', linewidth=0.75, marker='x', label="Sum Err. 1355")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 1), control_errors[n_subjects:,0],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 678")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 2), control_errors[n_subjects:,1],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 1355")

    ax.set(xticks=[1,2], xticklabels=[str(x) for x in [678,1355]], ylim=[-0.1, 1], xlim=[0,3])
    ax.set_title(f"Summed MAE from Control Patterns", fontsize=14, fontfamily='serif')
    ax.set_xlabel("Control Test Pattern", fontsize=12, fontfamily='serif')
    ax.set_ylabel("Summed MAE from Mean Tapped Pattern", fontsize=12, fontfamily='serif')
    ax.axhline(y=0,color='black', alpha=0.6, linestyle='--')



    ax1.scatter(control_errors[:,0], control_errors[:,1], marker='x', color='lightcoral')
    p1 = (outlier_thresholds[0],0.0)
    p2 = (0.0,outlier_thresholds[1])
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.scatter(np.mean(control_errors[:,0])+np.std(control_errors[:,0]), np.mean(control_errors[:,1])+np.std(control_errors[:,1]), marker='x', color='green', linewidth=0.8)
    ax1.text(outlier_thresholds[0],outlier_thresholds[1], str())
    ax1.set(xlim=[0,1], ylim=[0,1])
    for n in range(n_subjects):
        ax1.text(control_errors[n,0], control_errors[n,1], str(n+1), size='x-small' )
    ax1.set_title("Subject Control Pattern Cross Errors", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Summed MAE for Pattern 678", fontfamily='serif')
    ax1.set_ylabel("Summed MAE for Pattern 1355", fontfamily= 'serif')
    ax1.grid(color='lightgrey', linewidth=1, alpha=0.4)
    plt.show()

#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
### From here, we can begin to remove outliers that had a significantly larger error than the rest of the group.
### Step 1: Remove all participants that fail to tap within the accepted margin of error
n_subjects_clean=0
clean_subjects=[]
for person in range(len(by_person)):
    # Threshold check
    if control_errors[person][0]>=outlier_thresholds[0] or control_errors[person][1]>=outlier_thresholds[1]:
        print(f"[X]-->{person+1}") # remember idx!
    else:
        n_subjects_clean +=1
        clean_subjects.append(person)

by_person_clean=np.array([[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects_clean)], dtype=float) # [patt# tapresults]x18tests
print(f"Remaining subjects: {clean_subjects}")
clean_index=0

# Iterate through all participants
for person in range(len(by_person)):
    # Only copy results of non-outliers
    if person in clean_subjects: 
        #print(f"{person}")
        by_person_clean[clean_index]=by_person[person]
        clean_index += 1
control_errors = np.array([[0.0,0.0] for x in range(n_subjects_clean)], dtype=float)
for person in range(len(by_person_clean)):
    for test in range(len(by_person_clean[person])):
        m678 = np.mean(np.abs(by_person_clean[person][test][1:]-control_avgs[0]))
        m1355 = np.mean(np.abs(by_person_clean[person][test][1:]-control_avgs[1]))
        if by_person_clean[person][test][0]==controls[0]:
            #print(f"{person} {int(by_person_clean[person][test][0])} {m678}")
            control_errors[person][0] += m678
        if by_person_clean[person][test][0]==controls[1]:
            #print(f"{person} {by_person_clean[person][test][0]} {m1355}")
            control_errors[person][1] += m1355
    #print("----------")
### Plots for seeing outliers
if _controlcomparison:
    fig, (ax, ax1) = plt.subplots(1,2,figsize=(12,6))

    bp = ax.boxplot(control_errors, patch_artist=True, boxprops=dict(facecolor='none'))
    ax.scatter(np.full(len(control_errors), 1), control_errors[:,0],color='lightcoral', linewidth=0.75, marker='x', label="Sum Err. 678")
    ax.scatter(np.full(len(control_errors), 2), control_errors[:,1],color='lightcoral', linewidth=0.75, marker='x', label="Sum Err. 1355")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 1), control_errors[n_subjects:,0],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 678")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 2), control_errors[n_subjects:,1],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 1355")

    ax.set(xticks=[1,2], xticklabels=[str(x) for x in [678,1355]], ylim=[-0.1, 1], xlim=[0,3])
    ax.set_title(f"Summed MAE from Control Patterns", fontsize=14, fontfamily='serif')
    ax.set_xlabel("Control Test Pattern", fontsize=12, fontfamily='serif')
    ax.set_ylabel("Summed MAE from Mean Tapped Pattern", fontsize=12, fontfamily='serif')
    ax.axhline(y=0,color='black', alpha=0.6, linestyle='--')



    ax1.scatter(control_errors[:,0], control_errors[:,1], marker='x', color='lightcoral')
    p1 = (outlier_thresholds[0],0.0)
    p2 = (0.0,outlier_thresholds[1])
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.scatter(np.mean(control_errors[:,0])+np.std(control_errors[:,0]), np.mean(control_errors[:,1])+np.std(control_errors[:,1]), marker='x', color='green', linewidth=0.8)
    ax1.text(outlier_thresholds[0],outlier_thresholds[1], str())
    ax1.set(xlim=[0,1], ylim=[0,1])
    for n in range(n_subjects_clean):
        ax1.text(control_errors[n,0], control_errors[n,1], str(n+1), size='x-small' )
    ax1.set_title("Subject Control Pattern Cross Errors", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Summed MAE for Pattern 678", fontfamily='serif')
    ax1.set_ylabel("Summed MAE for Pattern 1355", fontfamily= 'serif')
    ax1.grid(color='lightgrey', linewidth=1, alpha=0.4)
    plt.show()
#----------------------------------------------------------------------------------------------------------------#

### Step 2: Remove the second control test for both (678,1355), sort into by pattern.
by_person_final=np.array([[[0.0 for x in range(17)] for y in range(16)] for z in range(n_subjects_clean)], dtype=float) # [patt# tapresults]x16tests
for person in range(len(by_person_clean)):
    ctrl1_cnt = 0
    ctrl2_cnt = 0
    test_cnt = 0
    
    for test in range(len(by_person_clean[person])):
        test_number = int(by_person_clean[person][test][0])
        if test_number==678:
            if ctrl1_cnt==0:
                by_person_final[person][test_cnt] = by_person_clean[person][test]
                ctrl1_cnt=1
                test_cnt+=1
                #print(f"{test_number} ctrl1 accepted - {test_cnt} = test cnt")
            else:
                #print(f"{test_number} ctrl1 rejected - {test_cnt} = test cnt")
                ctrl1_cnt=ctrl1_cnt
        elif test_number==1355:
            if ctrl2_cnt==0:
                by_person_final[person][test_cnt] = by_person_clean[person][test]
                ctrl2_cnt=1
                test_cnt+=1
                #print(f"{test_number} ctrl2 accepted - {test_cnt} = test cnt")
            else:
                #print(f"{test_number} ctrl2 rejected - {test_cnt} = test cnt")
                ctrl1_cnt=ctrl1_cnt
        elif test_number!=1355 and test_number!=678:
            one = by_person_clean[person][test]
            by_person_final[person][test_cnt] = one #by_person_clean[person][test]
            test_cnt+=1
            #print(f"here {test_number} - {test_cnt} = test cnt")
        #print(f"Person {person} - {test}|{test_cnt}-[{int(by_person_clean[person][test][0])}] {ctrl1_cnt}-{ctrl2_cnt}")
    #print('\n')
#----------------------------------------------------------------------------------------------------------------#

### Step 3: Recalculate means, only take patterns that are used (not 2nd control)
by_pattern = np.array([[[0.0 for x in range(16)] for y in range(n_subjects_clean)] for z in range(len(test_patterns))])
mean_diff = np.array([[[0.0 for x in range(16)] for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
mean_diff_org = np.array([[[0.0 for x in range(16)] for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
mean_diff_box = np.array([[0.0 for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
patt_mean_diff_box = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(len(test_patterns))])

# Sort into patterns.
for person in range(len(by_person_final)): # also = n_subjects_clean
    for test in range(len(by_person_final[person])): # 16
        for patt in range(len(test_patterns)): 
            if by_person_final[person][test][0]==test_patterns[patt]:
                by_pattern[patt][person] = by_person_final[person][test][1:]

# Calculate means
patt_means = [[0.0 for x in range(16)] for x in range(16)]
patt_stds = [[0.0 for x in range(16)] for x in range(16)]
patt_vars = [[0.0 for x in range(16)] for x in range(16)]
for patt in range(len(by_pattern)):
    patt_means[patt] = np.mean(by_pattern[patt], axis=0)
    patt_stds[patt] = np.std(by_pattern[patt], axis=0)
    patt_vars[patt] = np.var(by_pattern[patt], axis=0)

# Sort mean differences by person, and by pattern
for person in range(len(by_person_final)): # also = n_subjects_clean
    for test in range(len(by_person_final[person])): # 16
        for patt in range(len(test_patterns)): 
            if by_person_final[person][test][0]==test_patterns[patt]:
                mean_diff[person][test] =  np.abs(by_person_final[person][test][1:] - patt_means[patt])
                mean_diff_org[person][patt] =  np.abs(by_person_final[person][test][1:] - patt_means[patt])
        mean_diff_box[person][test]=np.mean(mean_diff[person][test])
        patt_mean_diff_box[test][person]=np.mean(mean_diff[person][test])


### Step 4: Calculate results for the initial tapping calibration test. 
for line in range(len(taptap)):
    for tap in range(32):
        taptap[line][tap+3]==float(taptap[line][tap+3])
taps = [[[0.0 for x in range(32)] for y in range(3)] for z in range(n_subjects_clean)]
taps_mid = [[0.0 for x in range(32)] for x in range(n_subjects_clean)]
taps_high = [[0.0 for x in range(32)] for x in range(n_subjects_clean)]
taps_low = [[0.0 for x in range(32)] for x in range(n_subjects_clean)]
count=0
mean_high=np.array([0.0 for x in range(32)], dtype=float)
mean_mid=np.array([0.0 for x in range(32)], dtype=float)
mean_low=np.array([0.0 for x in range(32)], dtype=float)

# by participant
for line in range(n_subjects):
    for subj in range(len(clean_subjects)):
        if clean_subjects[subj]==line: # subject id
            for n in range(3):
                index = line*3 + n
                test = int(taptap[index][2])
                if test==1:
                    taps_low[subj]=taptap[index][3:]
                    taps[subj][test-1]=taptap[index][3:]
                    mean_low += np.array(taptap[index][3:], dtype=float)
                if test==3:
                    taps_mid[subj]=taptap[index][3:]
                    taps[subj][test-1]=taptap[index][3:]
                    mean_mid += np.array(taptap[index][3:], dtype=float)
                if test==2:
                    taps_high[subj]=taptap[index][3:]
                    taps[subj][test-1]=taptap[index][3:]
                    mean_high += np.array(taptap[index][3:], dtype=float)

mean_high /=n_subjects_clean
mean_mid /=n_subjects_clean
mean_low /=n_subjects_clean


#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

### Plotting and analysis

### GENERAL ERROR PER SUBJECT
### (X: Subject, Y: MAE)
# How do people's abs. mean error compare to the mean tapped velocity for that pattern?
# --> Tells us how the good tappers performed overall.
if _subjectaverageerror:
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    idx = np.array(np.arange(n_subjects_clean))
    labels = idx+1
    custom_colors = mpl.cm.get_cmap('tab20b', n_subjects_clean)

    ax.axhline(y=0,color='black', alpha=0.8, linewidth=1,label='No Error')
    for n in range(n_subjects_clean):
        ax.scatter(np.full(len(mean_diff_box[n]), n+1), mean_diff_box[n], color=custom_colors.colors[n], linestyle='-', marker='x', label="Subject Avg. Err.")
    bp=ax.boxplot(mean_diff_box.T, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    for box, color in zip(bp['boxes'], custom_colors.colors):
        box.set_alpha=0.4
        box.set_facecolor(color)
    
    ax.set_title(f"Subject v. Subject Mean Abs. Error (all patts)", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Subject #", fontsize=13, fontfamily='sans-serif')
    ax.set_ylabel("MAE in tap velocity over all patterns", fontsize=13, fontfamily='sans-serif')
    ax.set(xticks=idx+1, xticklabels=[str(x) for x in labels], ylim=(-0.1,1))

    plt.show()

### GENERAL ERROR PER PATTERN
### (X: Pattern, Y: MAE)
# How do pattern's abs. mean error compare to the mean tapped velocity for all patterns?
# --> Tells us about where certain patterns were more or less problematic to reproduce.
if _patternaverageerror:
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot()
    pidx=np.arange(16)+1
    custom_colors_patterns = mpl.cm.get_cmap('viridis',24) # 24 to avoid yellows!
    
    ax.axhline(y=0,color='black', alpha=0.6)
    for i in range(len(test_patterns)):
        a = np.min([(float(i/len(test_patterns))+0.2),1])
        ax.scatter(np.full(len(patt_mean_diff_box[i]),i+1),patt_mean_diff_box[i], color=custom_colors_patterns.colors[i], linewidth=0.5, marker='x', alpha=a)
    bp= ax.boxplot(patt_mean_diff_box.T,widths=0.4, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    for box, color in zip(bp['boxes'], custom_colors_patterns.colors):
        box.set_alpha=0.5
        box.set_facecolor(color)
    
    ax.set(xticks=pidx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.1,1))
    ax.set_title(f"Mean Velocity Error from Mean by Pattern", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Test Pattern", fontsize=12)
    ax.set_ylabel("Mean Velocity Error (tap velocity)", fontsize=12, fontfamily='serif')
    plt.show()
    print()


### GENERAL ERROR VS CALIBRATION TAPS
### (X: Calibration Taps, Y: Tapped Value) (L/M/H tests on same graph)
# How did subjects tap when aiming for a target range
if _tapcalibration:
    tapidx=np.arange(32) 
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    for k in range(n_subjects_clean):
        ax.scatter(tapidx+1,np.array(taps_low[k], dtype=float), color='lightcoral', marker='x', linestyle='--', linewidth=1)
        ax.scatter(tapidx+1,np.array(taps_mid[k], dtype=float), color='lightgreen', marker='x', linestyle='--', linewidth=1)
        ax.scatter(tapidx+1,np.array(taps_high[k], dtype=float), color='lightblue', marker='x', linestyle='--', linewidth=1)

    ax.plot(tapidx+1, mean_high, color='blue', label='Mean Tap High')
    """ for n in range(3):
        print(taptap[((7*3))+n][1])
    ax.plot(tapidx+1, np.array(taps_high[7],dtype=float), color='indigo', label='Mean Tap High') """
    b=ax.boxplot(np.array(taps_high,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, facecolor='lightblue'))
    ax.plot(tapidx+1, mean_mid, color='green', label='Mean Tap Mid')
    b2=ax.boxplot(np.array(taps_mid,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, facecolor='lightgreen'))
    b3=ax.boxplot(np.array(taps_low,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, facecolor='lightcoral'))
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


#### TODO: 

### SUBJECT V CALIBRATION TAP ERROR
## (X: Subject, Y: MAE for target range (middle of range?)), 3 datapoints per subject
#
## ANOVA and Tukeys HSD for MAE in groups 
# By position: |0-50|51-100| , |0-33|34-66|67-100|, |0-25|26-50|51-75|76-100| (2/3/4)
# By test: (mid test should have least error, but show it)
# This will tell us if participants improved over the trials.

### PATTERN TAPPED ERROR v RHYTHM SPACE & GENRE



### GENERAL ERROR FOR SUBJECTS V ALGORITHMS