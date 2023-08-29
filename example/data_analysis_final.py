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
_subjectaverageerror = False
_patternaverageerror = False
_tapcalibration = True

####
n_subjects=25
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
outlier_thresholds = np.array([np.mean(control_errors[:,0])+(np.std(control_errors[:,0]*1.5)),np.mean(control_errors[:,1])+(np.std(control_errors[:,1])*1.5)], dtype=float)
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
    ax1.scatter(np.mean(control_errors[:,0])+(np.std(control_errors[:,0]*1.5)), np.mean(control_errors[:,1])+(np.std(control_errors[:,1]*1.5)), marker='o', color='green', linewidth=0.8)
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
outliers = []
outlier_control_errors = control_errors
for person in range(len(by_person)):
    # Threshold check
    if control_errors[person][0]>=outlier_thresholds[0] or control_errors[person][1]>=outlier_thresholds[1]:
        print(f"[X]-->{person+1}") # remember idx!
        outliers.append([control_errors[person][0], control_errors[person][1]])
    else:
        n_subjects_clean +=1
        clean_subjects.append(person)
outliers = np.array(outliers, dtype=float)
## *** ## 
## Adds outliers back in to analysis
_wholedataset = False
if _wholedataset:
    clean_subjects = np.arange(n_subjects,dtype=int)
    n_subjects_clean = n_subjects


by_person_clean=np.array([[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects_clean)], dtype=float) # [patt# tapresults]x18tests
print(f"Remaining subjects: {clean_subjects}")
clean_index=0

# Iterate through all participants
for person in range(len(by_person)):
    # Only copy results of non-outliers
    if person in clean_subjects: 
        by_person_clean[clean_index]=by_person[person]
        clean_index += 1
control_errors = np.array([[0.0,0.0] for x in range(n_subjects_clean)], dtype=float)
for person in range(len(by_person_clean)):
    for test in range(len(by_person_clean[person])):
        m678 = np.mean(np.abs(by_person_clean[person][test][1:]-control_avgs[0]))
        m1355 = np.mean(np.abs(by_person_clean[person][test][1:]-control_avgs[1]))
        if by_person_clean[person][test][0]==controls[0]:
            control_errors[person][0] += m678
        if by_person_clean[person][test][0]==controls[1]:
            control_errors[person][1] += m1355

### Step 2: Remove the second control test for both (678,1355), sort into by pattern.
by_person_final=np.array([[[0.0 for x in range(17)] for y in range(16)] for z in range(n_subjects_clean)], dtype=float) # [patt# tapresults]x16tests
for person in range(len(by_person_clean)):
    ctrl1_cnt = 0
    ctrl2_cnt = 0
    test_cnt = 0
    for test in range(len(by_person_clean[person])):
        test_number = int(by_person_clean[person][test][0])
        if test_number==678:
            #print(f"{person}[{test}] - {by_person_clean[person][test][0]} <----")
            if ctrl1_cnt==0:
                by_person_final[person][test_cnt] = by_person_clean[person][test]
                ctrl1_cnt=1
                test_cnt+=1
                ctrl1_cnt=ctrl1_cnt
        elif test_number==1355:
            #print(f"{person}[{test}] - {by_person_clean[person][test][0]} <----")
            if ctrl2_cnt==0:
                by_person_final[person][test_cnt] = by_person_clean[person][test]
                ctrl2_cnt=1
                test_cnt+=1
                ctrl1_cnt=ctrl1_cnt
        elif test_number!=1355 and test_number!=678:
            #print(f"{person}[{test}] - {by_person_clean[person][test][0]}")
            by_person_final[person][test_cnt] = by_person_clean[person][test]
            test_cnt+=1
    #print('\n')

### Step 3: Recalculate means, only take patterns that are used (not 2nd control)
by_pattern = np.array([[[0.0 for x in range(16)] for y in range(n_subjects_clean)] for z in range(len(test_patterns))])
mean_diff = np.array([[[0.0 for x in range(16)] for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
mean_diff_org = np.array([[[0.0 for x in range(16)] for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
mean_diff_box = np.array([[0.0 for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
mean_diff_raw_box = np.array([[0.0 for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
patt_mean_diff_box = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(len(test_patterns))])

# These two don't take the absolute error
mean_diff_raw = np.array([[[0.0 for x in range(16)] for y in range(len(test_patterns))] for z in range(n_subjects_clean)], dtype=float)
patt_mean_diff_raw = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(len(test_patterns))])

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
                mean_diff_raw[person][test] =  by_person_final[person][test][1:] - patt_means[patt]
                mean_diff_org[person][patt] =  np.abs(by_person_final[person][test][1:] - patt_means[patt]) #organized
                
        mean_diff_box[person][test]=np.mean(mean_diff[person][test])
        mean_diff_raw_box[person][test]=np.mean(mean_diff_raw[person][test])
        #mean_diff_raw[person][test]=np.mean(mean_diff_raw[person][test])
        patt_mean_diff_box[test][person]=np.mean(mean_diff[person][test])
        patt_mean_diff_raw[test][person]=np.mean(mean_diff_raw[person][test])


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
###                                             Plotting and analysis

if _controlcomparison:
    fig, (ax, ax1) = plt.subplots(1,2,figsize=(12,6))
    idx=np.array([1,2], dtype=float)
    bp = ax.boxplot(control_errors, positions=idx+0.1, patch_artist=True, boxprops=dict(facecolor='none'))
    bp2 = ax.boxplot(outlier_control_errors, positions=idx-0.1, patch_artist=True, boxprops=dict(facecolor='none'))

    ax.scatter(np.full(len(control_errors), 1.1), control_errors[:,0],color='green', linewidth=0.75, marker='x', label="Clean Sum Err. 678")
    ax.scatter(np.full(len(control_errors), 2.1), control_errors[:,1],color='green', linewidth=0.75, marker='x', label="Clean Sum Err. 1355")

    ax.scatter(np.full(len(outlier_control_errors), 0.9), outlier_control_errors[:,0],color='lightcoral', linewidth=0.75, marker='x', label="All Sum Err. 678")
    ax.scatter(np.full(len(outlier_control_errors), 1.9), outlier_control_errors[:,1],color='lightcoral', linewidth=0.75, marker='x', label="All Sum Err. 1355")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 1), control_errors[n_subjects:,0],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 678")
    #ax.scatter(np.full(len(control_errors)-n_subjects, 2), control_errors[n_subjects:,1],color='lightblue', linewidth=0.75, marker='x', label="Sum Err. 1355")

    ax.set(xticks=[1,2], xticklabels=[str(x) for x in [678,1355]], ylim=[-0.1, 1], xlim=[0,3])
    ax.set_title(f"Summed MAE from Control Patterns", fontsize=14, fontfamily='serif')
    ax.set_xlabel("Control Test Pattern", fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel("Summed MAE from Mean Tapped Pattern", fontsize=12, fontfamily='sans-serif')
    ax.axhline(y=0,color='black', alpha=0.6, linestyle='--')
    ax.legend(prop={'size':8})


    ax1.scatter(control_errors[:,0], control_errors[:,1], marker='x', color='green', label="Remaining Subjects")
    ax1.scatter(outliers[:,0], outliers[:,1], marker='x', color='lightcoral', label="Outliers")
    p1 = (outlier_thresholds[0],0.0)
    p2 = (0.0,outlier_thresholds[1])
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax1.add_patch(rect)

    ax1.scatter(outlier_thresholds[0], outlier_thresholds[1], marker='o', color='green', linewidth=0.8, label="Outlier Threshold")

    #ax1.text(outlier_thresholds[0],outlier_thresholds[1], str())
    ax1.set(xlim=[0,1], ylim=[0,1])
    for n in range(n_subjects_clean):
        ax1.text(control_errors[n,0], control_errors[n,1], str(n+1), size='x-small')
    ax1.set_title("Subject Control Pattern Cross Errors", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Summed MAE for Pattern 678", fontfamily='sans-serif', fontsize=12)
    ax1.set_ylabel("Summed MAE for Pattern 1355", fontfamily= 'sans-serif', fontsize=12)
    ax1.grid(color='lightgrey', linewidth=1, alpha=0.4)
    ax1.legend(prop={'size':8}, loc= "upper left")
    plt.show()


#------------------------------------------------------#
#------------------------------------------------------#
### GENERAL ERROR PER SUBJECT
### (X: Subject, Y: MAE)
# How do people's abs. mean error compare to the mean tapped velocity for that pattern?
# --> Tells us how the good tappers performed overall.
if _subjectaverageerror:

#--1--#
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(12,8))
    #ax = fig.add_subplot()
    idx = np.array(np.arange(n_subjects_clean))
    labels = idx+1
    custom_colors = mpl.cm.get_cmap('tab20b', n_subjects_clean)

    ax.axhline(y=0,color='black', alpha=0.8, linewidth=1,label='Mean Tapped Velocity', linestyle='--')
    for n in range(n_subjects_clean):
        #print(f"{len(mean_diff_box[n])} {mean_diff_raw[n]} ")
        ax.scatter(np.full(len(mean_diff_box[n]), n+1), mean_diff_box[n], color=custom_colors.colors[n], linestyle='-', marker='x', label="Subject Mean Abs. Err.", linewidth=0.8)
    bp=ax.boxplot(mean_diff_box.T, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    for box, color in zip(bp['boxes'], custom_colors.colors):
        box.set_alpha=0.4
        box.set_facecolor(color)
    
    ax.set_title(f"Subject v. Subject Mean Abs. Error (all patts)", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Subject #", fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel("MAE in tap velocity over all patterns", fontsize=12, fontfamily='sans-serif')
    ax.set(xticks=idx+1, xticklabels=[str(x) for x in labels], ylim=(-0.1,0.6))
    #plt.show()

#--2--#
    #fig = plt.figure(figsize=(12,6))
    #ax1 = fig.add_subplot()
    idx = np.array(np.arange(n_subjects_clean))
    labels = idx+1
    custom_colors = mpl.cm.get_cmap('tab20b', n_subjects_clean)

    ax1.axhline(y=0,color='black', alpha=0.8, linewidth=1,label='Mean Tapped Velocity', linestyle='--')
    for n in range(n_subjects_clean):
        #print(f"{len(mean_diff_raw[n])} {mean_diff_raw[n]} ")
        ax1.scatter(np.full(len(mean_diff_raw_box[n]), n+1), mean_diff_raw_box[n], color=custom_colors.colors[n], linestyle='-', marker='x', label="Subject Avg. Err.", linewidth=0.8)
    bp=ax1.boxplot(mean_diff_raw_box.T, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    for box, color in zip(bp['boxes'], custom_colors.colors):
        box.set_alpha=0.4
        box.set_facecolor(color)
    
    ax1.set_title(f"Non-Outlier Subjects v. Subject Mean Error (all patts)", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Subject #", fontsize=12, fontfamily='sans-serif')
    ax1.set_ylabel("Mean Error in tap velocity over all patterns", fontsize=12, fontfamily='sans-serif')
    ax1.set(xticks=idx+1, xticklabels=[str(x) for x in labels], ylim=(-0.4,0.5))

    fig.tight_layout()
    plt.show()


#------------------------------------------------------#
#------------------------------------------------------#
### GENERAL ERROR PER PATTERN
### (X: Pattern, Y: MAE)
# How do pattern's abs. mean error compare to the mean tapped velocity for all patterns?
# --> Tells us about where certain patterns were more or less problematic to reproduce.
if _patternaverageerror:

#--1--#
    fig, (ax, ax1) = plt.subplots(2,1,figsize=(12,8))
    #ax = fig.add_subplot()
    pidx=np.arange(16)+1
    custom_colors_patterns = mpl.cm.get_cmap('cividis',96) # 24 to avoid yellows!
    color='darkslategray' #custom_colors_patterns.colors[i]

    ax.axhline(y=0,color='black', alpha=0.6, linestyle='--', label='Mean Tapped Velocity')
    for i in range(len(test_patterns)):
        #a = np.min([(float(i/len(test_patterns))+0.2),1])
        a=0.7
        ax.scatter(np.full(len(patt_mean_diff_box[i]),i+1),patt_mean_diff_box[i], color=color, linewidth=0.5, marker='x', alpha=a)
    bp= ax.boxplot(patt_mean_diff_box.T,widths=0.4, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    for box, color in zip(bp['boxes'], custom_colors_patterns.colors):
        box.set_alpha=0.5
        box.set_facecolor(color)
    
    ax.set(xticks=pidx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.1,0.6))
    ax.set_title(f"Mean Absolute Error from Subject Tapped Patterns", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Test Pattern", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (tap velocity)", fontsize=12, fontfamily='serif')

#--2--#
    pidx=np.arange(16)+1
    ax1.axhline(y=0,color='black', alpha=0.6, linestyle='--', label='Mean Tapped Velocity')    
    for i in range(len(test_patterns)):
        #a = np.min([(float(i/len(test_patterns))+0.2),1])
        a=0.7
        ax1.scatter(np.full(len(patt_mean_diff_raw[i]),i+1),patt_mean_diff_raw[i], color=color, linewidth=0.5, marker='x', alpha=a)
    bp= ax1.boxplot(patt_mean_diff_raw.T,widths=0.4, patch_artist=True, boxprops=dict(alpha=0.3, facecolor='lightcoral', edgecolor='black'))
    for box, color in zip(bp['boxes'], custom_colors_patterns.colors):
        box.set_alpha=0.5
        box.set_facecolor(color)
    
    ax1.set(xticks=pidx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.4,0.5))
    ax1.set_title(f"Mean Error from Subject Tapped Patterns", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Test Pattern", fontsize=12, fontfamily='sans-serif')
    ax1.set_ylabel("Mean Error (tap velocity)", fontsize=12, fontfamily='sans-serif')
    fig.tight_layout()
    plt.show()


#------------------------------------------------------#
#------------------------------------------------------#
### GENERAL ERROR VS CALIBRATION TAPS
### (X: Calibration Taps, Y: Tapped Value) (L/M/H tests on same graph)
# How did subjects tap when aiming for a target range
if _tapcalibration:
#--1--#
    # Tap Order v LMH Mean Tap
    tapidx=np.arange(32) 
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot()

    colors_light=['mediumpurple','lightblue','lightgreen']
    colors_dark=['indigo','royalblue','forestgreen']


    for k in range(n_subjects_clean):
        ax.scatter(tapidx+1,np.array(taps_low[k], dtype=float), color=colors_light[2], marker='x', linestyle='--', linewidth=1)
        ax.scatter(tapidx+1,np.array(taps_mid[k], dtype=float), color=colors_light[1], marker='x', linestyle='--', linewidth=1)
        ax.scatter(tapidx+1,np.array(taps_high[k], dtype=float), color=colors_light[0], marker='x', linestyle='--', linewidth=1)

    b=ax.boxplot(np.array(taps_high,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, facecolor=colors_light[0]))
    b2=ax.boxplot(np.array(taps_mid,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, facecolor=colors_light[1]))
    b3=ax.boxplot(np.array(taps_low,dtype=float), patch_artist=True, boxprops=dict(linewidth=1, alpha=0.3, facecolor=colors_light[2]))
    ax.plot(tapidx+1, mean_low, color=colors_dark[2], label='Mean Tap Low')
    ax.plot(tapidx+1, mean_mid, color=colors_dark[1], label='Mean Tap Mid')
    ax.plot(tapidx+1, mean_high, color=colors_dark[0], label='Mean Tap High')

    ax.axhline(y=42,color='seagreen', alpha=0.6, linestyle='--', label='SOFT/MID Boundary')
    ax.axhline(y=84,color='slateblue', alpha=0.6, linestyle='--', label='MID/HARD Boundary')

    ax.set(ylim=[0,127], yticks=[0,42,84,127], xticks=tapidx+1)
    ax.set_title(f"Progressive Tapping Consistency over all Subjects", fontsize=14, fontfamily='serif',fontweight='book')
    ax.set_xlabel("Tap Order", fontsize=12, fontfamily='sans-serif')
    ax.set_ylabel("Tapped Velocity", fontsize=12, fontfamily='sans-serif')
    plt.legend(loc='upper left', bbox_to_anchor=(-0.15, 0.95),prop={'size': 8})
    plt.show()

#--2--#
    # Subject v MAE to middle of target per test.
    fig= plt.figure(figsize=(12,6))
    ax1=fig.add_subplot()
    subj_idx = np.arange(n_subjects_clean)+1

    taps_low = np.array(taps_low, dtype=float)
    taps_mid = np.array(taps_mid, dtype=float)
    taps_high = np.array(taps_high, dtype=float)
    
    low_mean = np.mean(np.mean(np.array(taps_low, dtype=float)))
    mid_mean = np.mean(np.mean(np.array(taps_mid, dtype=float)))
    high_mean = np.mean(np.mean(np.array(taps_high, dtype=float)))

    _taps_mid = np.array([[0.0 for x in range(32)] for x in range(n_subjects_clean)], dtype=float)
    _taps_high = np.array([[0.0 for x in range(32)] for x in range(n_subjects_clean)], dtype=float)
    _taps_low = np.array([[0.0 for x in range(32)] for x in range(n_subjects_clean)], dtype=float)

    ax1.grid(color='lightgray', linestyle='-', linewidth=0.6, alpha=0.7, axis='y')

    pos_low = subj_idx - 0.25
    pos_high = subj_idx + 0.25
    width = 0.20

    line0=ax1.axhline(y=0,color='black', alpha=0.6, linestyle='--', label='Middle of Target Range')
    p1 = (n_subjects_clean+1,0.0)
    p2 = (0.0,float(0.33/2))
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=1, edgecolor='darkgreen', facecolor='lightgreen', alpha=0.15, linestyle='--', label='Target Range (+/-)')
    ax1.add_patch(rect)

    for k in range(n_subjects_clean):
        _taps_mid[k]= np.abs((taps_mid[k]-63))/127
        _taps_low[k]= np.abs((taps_low[k]-21))/127
        _taps_high[k]= np.abs((taps_high[k]-105))/127

    flierprops=dict(marker='x',color=colors_light[0], markersize='8')
    bp1 = ax1.boxplot(_taps_low.T, positions=pos_low, widths=width, patch_artist=True, showfliers=True, boxprops=dict(facecolor=colors_light[2], alpha=0.5),flierprops=dict(marker='x',markeredgecolor=colors_light[2], markersize='5'))
    bp2 = ax1.boxplot(_taps_mid.T, positions=subj_idx, widths=width, patch_artist=True, showfliers=True, boxprops=dict(facecolor=colors_light[1], alpha=0.5,),flierprops=dict(marker='x',markeredgecolor=colors_light[1], markersize='5'))
    bp3 = ax1.boxplot(_taps_high.T, positions=pos_high, widths=width, patch_artist=True, showfliers=True, boxprops=dict(facecolor=colors_light[0], alpha=0.5),flierprops=dict(marker='x',markeredgecolor=colors_light[0], markersize='5')) 

    for k in range(n_subjects_clean):
        ax1.plot(pos_low[k], np.mean(np.abs(taps_low[k]-21))/127, color=colors_dark[2], marker='x', alpha=1)
        ax1.plot(subj_idx[k], np.mean(np.abs(taps_mid[k]-63))/127, color=colors_dark[1], marker='x', alpha=1)
        ax1.plot(pos_high[k], np.mean(np.abs(taps_high[k]-105))/127, color=colors_dark[0], marker='x', alpha=1)
    
    ax1.set(xlim=[0,n_subjects_clean+1], ylim=[-0.06,1.0], xticks=subj_idx, xticklabels=subj_idx, yticks=np.arange(start=0.0,stop=1.0,step=0.1))
    ax1.set_title(f"Subject Mean Abs. Err. for Tap Consistency", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_xlabel("Subject #", fontsize=12, fontfamily='sans-serif')
    ax1.set_ylabel("Mean Tapped Error", fontsize=12, fontfamily='sans-serif')

    line1, = ax1.plot([1,1], color='blue')
    line2, = ax1.plot([1,1], color='green')
    line3, = ax1.plot([1,1], color='red')

    # Add the legend
    #ax.legend((line1, line2, line3), ('data1', 'data2', 'data3'), loc='upper right')

    # Remove the dummy lines
    line1.set_visible(False)
    line2.set_visible(False)
    line3.set_visible(False)

    ax1.legend([bp1["boxes"][0], bp2["boxes"][0],bp3["boxes"][0], rect, line0], ('Soft Tap Range', 'Mid Tap Range', 'Hard Tap Range', 'Target Range (+/-)', 'Middle of Target Range'))
    fig.tight_layout()
    plt.show()

## ANOVA and Tukeys HSD for MAE in groups 
# By position: |0-50|51-100| , |0-33|34-66|67-100|, |0-25|26-50|51-75|76-100| (2/3/4)
# By test: (mid test should have least error, but show it). This will tell us if participants improved over the trials.

    _printANOVA = False
    f_stat,p_val = stats.f_oneway(_taps_high,_taps_mid,_taps_low)
    if _printANOVA:
        print(f_stat)
        print(p_val)
        print(f"Tap Tap by Range ANOVA: ")
        #print(f"F-Statistics: {f_stat:1.4f}")
        #print(f"P-Values: {p_val:1.6f}")
    # Tukey's HSD
    _tukey = True
    if _tukey:
        # Reorder data for tukey
        pre_df=[]
        for j in range(len(_taps_high)):
            pre_df.append([0, _taps_high[j]])
        for j in range(len(_taps_mid)):
            pre_df.append([1, _taps_mid[j]])
        for j in range(len(_taps_low)):
            pre_df.append([2, _taps_low[j]])

        # Do Tukey's HSD
        test_type=[row[0] for row in pre_df]
        dist_val = [row[1] for row in pre_df]
        df = pd.DataFrame({'test_type': test_type, 'distance':dist_val})
        tukey = pairwise_tukeyhsd(endog=df["distance"],groups=df['test_type'], alpha=0.05)
        np.set_printoptions(precision=6)
        #print(tukey)
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

        # Set the desired precision for p-adj values
        pd.set_option('display.float_format', '{:.6f}'.format)

        # Print the DataFrame
        if _printANOVA:
            print(tukey_df)

#### TODO: 

### SUBJECT ERROR (NOT MAE) vs ALL PATTERNS [>>>>>>>>>>>>>>>]
### PATTERN ERROR (NOT MAE) vs ALL SUBJECTS [>>>>>>>>>>>>>>>]
# Allows us to see direction of errors

### SUBJECT V CALIBRATION TAP ERROR
## (X: Subject, Y: MAE for target range (middle of range?)), 3 datapoints per subject [>>>>>>>>>>>>>>>]



### PATTERN TAPPED ERROR v RHYTHM SPACE & GENRE

### GENERAL ERROR FOR SUBJECTS V ALGORITHMS
## ANOVA and Tukeys HSD for MAE of Algorithms vs Subjects
# Gives us comparison for which algorithm best reflected subjects & was sign. better than others
# ------> Needs to be done with cleaned data!
'''
mse = np.array([0.0 for x in range(6)], dtype=float)
mae = np.array([0.0 for x in range(6)], dtype=float)
rmse = np.array([0.0 for x in range(6)], dtype=float)
rsqr = np.array([0.0 for x in range(6)], dtype=float)
mape = np.array([0.0 for x in range(6)], dtype=float)
alg_scores = np.array([[0.0 for x in range(6)] for x in range(5)])
for i in range(len(avgs)):
    # Can limit to max of tapped inputs
    # algs[i] = np.where(algs[i]<np.max(avgs[i]),algs[i],np.max(avgs[i]))
    for j in range(len(algs[0])):
        mse[j]+=sk.mean_squared_error(avgs[i], algs[i][j])
        mae[j]+=sk.mean_absolute_error(avgs[i], algs[i][j])
        rsqr[j]+=sk.r2_score(avgs[i], algs[i][j])
        rmse[j]+=pow(np.abs(sk.mean_squared_error(avgs[i], algs[i][j])),0.5)
        mape[j]+=sk.mean_absolute_percentage_error(avgs[i],algs[i][j])

# Avg out error readings
mae /= 16
alg_scores[0]=mae
mse /= 16
alg_scores[1]=mse
rmse /= 16
alg_scores[2]=rmse
rsqr /= 16
alg_scores[3]=rsqr
mape /= 16
alg_scores[4]=mape
test_names = ["mae","mse","rmse","rsqr","mape"]
_printtest = True
if _printtest:
    # Print error-test results
    for tn in range(len(test_names)): # iterate through test types
        print(f"{test_names[tn]}:-------------:")
        for algtype in range(len(alg_names)): # iterate through 6 alg types
            print(f"{alg_scores[tn][algtype]:1.4f} <- {alg_names[algtype]}")


_printANOVA = True
# ANOVA
f_stat,p_val = stats.f_oneway(data_anova[0],data_anova[1],data_anova[2],data_anova[3],data_anova[4],data_anova[5])
if _printANOVA:
    print(f"{test_patterns[i]} ANOVA: ")
    print(f"F-Statistics: {f_stat:1.4f}")
    print(f"P-Values: {p_val:1.6f}")

# Tukey's HSD
_tukey = True
if _tukey:
    # Reorder data for tukey
    pre_df=[]
    for i in range(6):
        #print(f"len d_anova {len(data_anova[i])}")
        for j in range(len(data_anova[i])):
            pre_df.append([i, data_anova[i][j]])

    # Do Tukey's HSD
    alg_val=[row[0] for row in pre_df]
    dist_val = [row[1] for row in pre_df]
    df = pd.DataFrame({'algorithm': alg_val, 'distance':dist_val})
    tukey = pairwise_tukeyhsd(endog=df["distance"],groups=df['algorithm'], alpha=0.05)
    np.set_printoptions(precision=6)
    #print(tukey)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    # Set the desired precision for p-adj values
    pd.set_option('display.float_format', '{:.6f}'.format)

    # Print the DataFrame
    if _printANOVA:
        print(tukey_df)

        
'''
###  