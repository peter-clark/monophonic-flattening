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
import neuralnetwork as NN
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib\..*") # :)

################################## Data Analysis for Tap to Drums ###################################
# Before running analysis, give number of participants and which graphs/analyses to do:
n_subjects=43
_wholedataset = False

# Which analysis
_coordinates = False
_control = True # must be true for some reason not yet found in the plot block
_subject = False
_pattern = False
_position = True
_tapcalibration = True

######################################################################################################
########################## Variable declaration and subject data loading #############################
test_patterns = [894, 423, 1367, 249, 939, 427, 590, 143, 912, 580, 1043, 673, 1359, 736, 678, 1355]
control_patterns = [678, 1355]
algorithm_names = ["cont1", "disc1", "cont2", "disc2", "semicont1", "semicont2", "mbt"] #mbt is model-by-tap

# Main Experiment: Tap to Polyphonic Drum Pattern
data = []
tap_file = os.getcwd()+"/results/tapexplore.csv"
pickle_dir = os.getcwd()+"/data/"
with open(tap_file) as results: 
    reader = csv.reader(results)
    for row in reader:
        data.append(row)
    results.close()

# Secondary Experiment: Tapping Consistency
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


######################################################################################################
################################## Main Analyses Data Structures #####################################
algorithms = np.array([[[0.0 for x in range(16)] for y in range(16)] for z in range(7)], dtype=float) # 6 algs + mbt
subjects = np.array([[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects)], dtype=float) #[patt#(1), testresults(16)]
controls = np.array([[[0.0 for x in range(16)] for y in range(4)] for z in range(n_subjects)], dtype=float) # 16steps x 4(2x2) x N
ctrl_cnt = [0,0]
print(f"{len(data)%18} & {len(data)}/18={(len(data)-1)/18}") # Data check

subject_count = 0
test_count = 0

## Sort into data structs. ##
## Go through every test and sort. ##
for test in range(len(data)):
    if test!=0: # skip header row
        patt_number = int(data[test][2])

        # Organize 5th/6th Algorithm
        sc = np.array([[0.0 for x in range(16)] for y in range(2)], dtype=float)
        alg_hold = np.array([[0.0 for x in range(16)] for y in range(6)], dtype=float)
        for i in range(16):
            sc[0][i]=float(data[test][20+i])*float(data[test][36+i]) #semicont1
            sc[1][i]=float(data[test][52+i])*float(data[test][68+i]) #semicont2
            data[test].append(sc[0][i])
        for i in range(16):
             data[test].append(sc[1][i])
        
        # Sort into by algorithm struct (can do now as predictions don't change)
        alg_hold[0] = np.asarray(data[test][20:36], dtype=float) #c1
        alg_hold[1] = np.asarray(data[test][36:52], dtype=float) #d1
        alg_hold[2] = np.asarray(data[test][52:68], dtype=float) #c2
        alg_hold[3] = np.asarray(data[test][68:84], dtype=float) #d2
        alg_hold[4] = np.asarray(data[test][-32:-16], dtype=float) #sc1
        alg_hold[5] = np.asarray(data[test][-16:], dtype=float) #sc2
        for i in range(len(test_patterns)):
            if test_patterns[i] == patt_number:
                for j in range(len(alg_hold)):
                    algorithms[j][i] = alg_hold[j]

        # Sort into by subject struct (to be later cleaned)
        tap_data = np.array([0.0 for x in range(17)]) # name + tap
        tap_data[0] = patt_number
        tap_data[1:] = np.asarray([data[test][4:20]], dtype=float)

        # Sort if controls
        if patt_number == control_patterns[0]: # 678
            controls[subject_count][ctrl_cnt[0]%2] = tap_data[1:]
            ctrl_cnt[0]+=1
        if patt_number == control_patterns[1]: # 1355
            controls[subject_count][(ctrl_cnt[1]%2)+2] = tap_data[1:]
            ctrl_cnt[1]+=1
        
        # Roll over to next subject if at end
        test_number = test-1
        if test_number%18==0 and test_number!=0: 
            subject_count += 1
            test_count = 0

        # Save tap data to subjects
        if subject_count != n_subjects:
            subjects[subject_count][test_count] = tap_data
            test_count += 1

## More variables
taptaps = np.array([[[0.0 for x in range(32)] for y in range(3)] for z in range(n_subjects)], dtype=float)
taptaps_mid = np.array([[0.0 for x in range(32)] for x in range(n_subjects)], dtype=float)
taptaps_high = np.array([[0.0 for x in range(32)] for x in range(n_subjects)], dtype=float)
taptaps_low = np.array([[0.0 for x in range(32)] for x in range(n_subjects)], dtype=float)
mean_taptaps = np.array([[0.0 for x in range(32)] for x in range(3)], dtype=float)

## Sort tap consistency into by subject ##
for line in range(len(taptap)):
    for tap in range(32):
        taptap[line][tap+3]==float(taptap[line][tap+3])
print(mean_taptaps.shape)
for n in range(n_subjects):
    for t in range(3):
        index = n*3 + t
        test = int(taptap[index][2])
        if test==1:
            taptaps_low[n]=taptap[index][3:]
            taptaps[n][test-1]=taptap[index][3:]
            mean_taptaps[0] += np.array(taptap[index][3:], dtype=float)
        if test==3:
            taptaps_mid[n]=taptap[index][3:]
            taptaps[n][test-1]=taptap[index][3:]
            mean_taptaps[1] += np.array(taptap[index][3:], dtype=float)
        if test==2:
            taptaps_high[n]=taptap[index][3:]
            taptaps[n][test-1]=taptap[index][3:]
            mean_taptaps[2] += np.array(taptap[index][3:], dtype=float)
mean_taptaps /= n_subjects           
for i in range(len(mean_taptaps)):
    print(f"Tap Consistency Means: {np.mean(mean_taptaps[i])}")

######################################################################################################
################################## Outlier Selection #################################################
## Variable Declaration ##
control_precision = np.asarray([[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)], dtype=float)
control_accuracy = np.asarray([[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)], dtype=float)
control_accuracy_abs = np.asarray([[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)], dtype=float)
control_means = np.asarray([[0.0 for x in range(16)] for y in range(2)], dtype=float)
outlier_thresholds = np.asarray([[0.0 for x in range(4)] for y in range(3)], dtype=float) # [678+, 678-, 1355+, 1355-] x 3 outlier tests

## Means for Accuracy Comparison ##
for n in range(n_subjects):
    control_means[0] += (controls[n][0] + controls[n][1])/2
    control_means[1] += (controls[n][2] + controls[n][3])/2
control_means[:] /= n_subjects
#print(control_means)
## Calculate Precision and Accuracy ##
for n in range(n_subjects):
    control_precision[n][0] = controls[n][0]-controls[n][1]
    control_precision[n][1] = controls[n][2]-controls[n][3]

    # POS/NEG refers to direction compared to overall subject mean!! 
    control_accuracy[n][0] = (controls[n][0]-control_means[0])+(controls[n][1]-control_means[0])
    control_accuracy[n][1] = (controls[n][2]-control_means[1])+(controls[n][3]-control_means[1])
    control_accuracy_abs[n][0] = np.abs(controls[n][0]-control_means[0])+np.abs(controls[n][1]-control_means[0])
    control_accuracy_abs[n][1] = np.abs(controls[n][2]-control_means[1])+np.abs(controls[n][3]-control_means[1])


## Precision Stats ##
precision_means = np.mean(control_precision, axis=2)
precision_means = np.asarray(precision_means, dtype=float)
print("PRECISION:\n678:")
print(f"[M={np.mean(np.abs(precision_means[:,0])):.3f}, SD={np.std(precision_means[:,0]):.3f}]")
print(f"[V={np.var(precision_means[:,0]):.3f}, CV={(np.std(precision_means[:,0])/np.mean(np.abs(precision_means[:,0]))):.3f}]")
print("1355:")
print(f"[M={np.mean(np.abs(precision_means[:,1]))}, SD={np.std(precision_means[:,1]):.3f}]")
print(f"[V={np.var(precision_means[:,1]):.3f}, CV={(np.std(precision_means[:,1])/np.mean(np.abs(precision_means[:,1]))):.3f}]")

## Accuracy Stats ##
ctrl_means = np.mean(control_accuracy, axis=2)
ctrl_means = np.asarray(ctrl_means, dtype=float)
print("\nACCURACY:\n678:")
print(f"[M={np.mean(np.abs(ctrl_means[:,0])):.3f}, SD={np.std(ctrl_means[:,0]):.3f}]")
print(f"[V={np.var(ctrl_means[:,0]):.3f}, CV={(np.std(ctrl_means[:,0])/np.mean(np.abs(ctrl_means[:,0]))):.3f}]")
print("1355:")
print(f"[M={np.mean(np.abs(ctrl_means[:,1]))}, SD={np.std(ctrl_means[:,1]):.3f}]")
print(f"[V={np.var(ctrl_means[:,1]):.3f}, CV={(np.std(ctrl_means[:,1])/np.mean(np.abs(ctrl_means[:,1]))):.3f}]")

ctrl_means_abs = np.mean(control_accuracy_abs, axis=2)
ctrl_means_abs = np.asarray(ctrl_means_abs, dtype=float)
print("\nABS ACCURACY:\n678:")
print(f"[M={np.mean(np.abs(ctrl_means_abs[:,0])):.3f}, SD={np.std(ctrl_means_abs[:,0]):.3f}]")
print(f"[V={np.var(ctrl_means_abs[:,0]):.3f}, CV={(np.std(ctrl_means_abs[:,0])/np.mean(np.abs(ctrl_means_abs[:,0]))):.3f}]")
print("1355:")
print(f"[M={np.mean(np.abs(ctrl_means_abs[:,1]))}, SD={np.std(ctrl_means_abs[:,1]):.3f}]")
print(f"[V={np.var(ctrl_means_abs[:,1]):.3f}, CV={(np.std(ctrl_means_abs[:,1])/np.mean(np.abs(ctrl_means_abs[:,1]))):.3f}]")

out = ctrl_means+precision_means #ctrl_means_abs
outlier_thresholds[1] = [np.std(ctrl_means[:,0])*2, -np.std(ctrl_means[:,0])*2, np.std(ctrl_means[:,1])*2, -np.std(ctrl_means[:,1])*2]
outlier_thresholds[2] = [np.std(ctrl_means_abs[:,0])*2, -np.std(ctrl_means_abs[:,0])*2, np.std(ctrl_means_abs[:,1])*2, -np.std(ctrl_means_abs[:,1])*2]

##---------------------Plot outliers----------------------##
if _control:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,8), gridspec_kw={'width_ratios': [2,2]}) 

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    ## Precision by Control Pattern ##
    p1=(np.std(precision_means[:,0])*2,-np.std(precision_means[:,1])*2)
    p2=(-np.std(precision_means[:,0])*2,np.std(precision_means[:,1])*2)

    outlier_thresholds[0] = [np.std(precision_means[:,0])*2, -np.std(precision_means[:,0])*2, np.std(precision_means[:,1])*2, -np.std(precision_means[:,1])*2]

    ax1.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax1.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.scatter(precision_means[:,0], precision_means[:,1], marker='.', color='darkgreen', label="Intra-Pattern Precision")
    for n in range(n_subjects): # Subject # Labels
        ax1.text(precision_means[n,0], precision_means[n,1], str(n+1), size='x-small')
    ax1.set(xlim=[-0.5,0.5], ylim=[-0.5,0.5], xticks=np.arange(-0.5,0.5,0.1), yticks=np.arange(-0.5,0.5,0.1))
    ax1.grid(color='lightgrey', linewidth=0.3, alpha=0.4)
    ax1.set_title("Precision (Attempt 1 v 2)", fontsize=14, fontfamily='serif',fontweight='book')
    ax1.set_ylabel("Pattern 678", fontsize=12, fontfamily='sans-serif')
    ax1.set_ylabel("Pattern 1355", fontsize=12, fontfamily='sans-serif')

    ## Accuracy by Control Pattern ##
    p1=(np.std(ctrl_means[:,0])*2,-np.std(ctrl_means[:,1])*2)
    p2=(-np.std(ctrl_means[:,0])*2,np.std(ctrl_means[:,1])*2)

    outlier_thresholds[1] = [np.std(ctrl_means[:,0])*2, -np.std(ctrl_means[:,0])*2, np.std(ctrl_means[:,1])*2, -np.std(ctrl_means[:,1])*2]

    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax2.add_patch(rect)
    ax2.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax2.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax2.scatter(ctrl_means[:,0], ctrl_means[:,1], marker='.', color='lightcoral', label="Sum Mean Accuracy")
    for n in range(n_subjects): # Subject # Labels
        ax2.text(ctrl_means[n,0], ctrl_means[n,1], str(n+1), size='x-small')
    ax2.set(xlim=[-1,1], ylim=[-1,1], xticks=np.arange(-1,1,0.25), yticks=np.arange(-1,1,0.25))
    ax2.grid(color='lightgrey', linewidth=0.3, alpha=0.4)
    ax2.set_title("True Accuracy (vs. Mean Tapped Pattern)", fontsize=14, fontfamily='serif',fontweight='book')
    ax2.set_ylabel("Pattern 678", fontsize=12, fontfamily='sans-serif')
    ax2.set_ylabel("Pattern 1355", fontsize=12, fontfamily='sans-serif')

    fig.legend()
    plt.show()

    ## Second Plot ##

    fig, (ax3, ax4) = plt.subplots(1,2, figsize=(12,8), gridspec_kw={'width_ratios': [2,2]}) 

    ax3.set_aspect('equal')
    ## Absolute Accuracy by Control Pattern ##
    p3=(np.mean(ctrl_means_abs[:,0])+(np.std(ctrl_means_abs[:,0])*2), np.mean(ctrl_means_abs[:,1])-(np.std(ctrl_means_abs[:,1])*2))
    p4=((np.mean(ctrl_means_abs[:,0])-(np.std(ctrl_means_abs[:,0]))*2), np.mean(ctrl_means_abs[:,1])+(np.std(ctrl_means_abs[:,1])*2))

    outlier_thresholds[2] = [np.std(ctrl_means_abs[:,0])*2, -np.std(ctrl_means_abs[:,0])*2, np.std(ctrl_means_abs[:,1])*2, -np.std(ctrl_means_abs[:,1])*2]

    rect = mpl.patches.Rectangle((p4[0], p3[1]), p3[0] - p4[0], p4[1] - p3[1], linewidth=0.8, edgecolor='blue', facecolor='none', linestyle='--')
    ax3.add_patch(rect)
    ax3.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax3.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax3.scatter(ctrl_means_abs[:,0], ctrl_means_abs[:,1], marker='.', color='lightblue', label="Sum Abs. Mean Accuracy")
    for n in range(n_subjects):
        ax3.text(ctrl_means_abs[n,0], ctrl_means_abs[n,1], str(n+1), size='x-small')
    ax3.set(xlim=[0,1], ylim=[0,1], xticks=np.arange(0,1,0.25), yticks=np.arange(0,1,0.25))
    ax3.grid(color='lightgrey', linewidth=0.3, alpha=0.4)
    ax3.set_title("Summed ABS Accuracy \n(vs. Mean Tapped Pattern)", fontsize=14, fontfamily='serif',fontweight='book')
    ax3.set_ylabel("Pattern 678", fontsize=12, fontfamily='sans-serif')
    ax3.set_ylabel("Pattern 1355", fontsize=12, fontfamily='sans-serif')

   
    ax4.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax4.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax4.scatter(out[:,0], out[:,1], marker='.', color='lightblue', label="Sum Abs. Mean Accuracy")
    for n in range(n_subjects):
        ax4.text(out[n,0], out[n,1], str(n+1), size='x-small')
    #ax4.set(xlim=[0,1], ylim=[0,1], xticks=np.arange(0,1,0.25), yticks=np.arange(0,1,0.25))
    ax4.grid(color='lightgrey', linewidth=0.3, alpha=0.4)
    ax4.set_title("Sum Outlier Metric", fontsize=14, fontfamily='serif',fontweight='book')
    ax4.set_ylabel("Pattern 678", fontsize=12, fontfamily='sans-serif')
    ax4.set_ylabel("Pattern 1355", fontsize=12, fontfamily='sans-serif')

    fig.legend()
    plt.show()    
    
######################################################################################################
########################## Sort cleaned subjects, patterns and steps #################################
## Variable Declaration ##
n_subjects_clean=0
subjects_clean_idx = []
outliers = []

## Identify outliers and clean subjects ##
for n in range(n_subjects):
    c = [np.mean(control_precision[n][0]), np.mean(control_precision[n][1])]
    if c[0]>=outlier_thresholds[0][0] or c[0]<=outlier_thresholds[0][1] or c[1]>=outlier_thresholds[0][2] or c[1]<=outlier_thresholds[0][3]: # 1355 check
            print(f"[x] {n+1} <--")
            outliers.append([n, c[0], c[1]])
    else:
        n_subjects_clean += 1
        print(f"[ ] {n+1}")
        subjects_clean_idx.append(n)
outliers=np.array(outliers, dtype=float)

## Analyze whole dataset if asked ##
if _wholedataset:
    subjects_clean_idx = np.arange(n_subjects, dtype=int)
    n_subjects_clean=n_subjects

print(f"Subjects: {n_subjects_clean} \nOutliers: {len(outliers[:,:1])} \n{outliers[:,:1]}")

## Declare new variable structs ##
subjects_full = np.array([[[0.0 for x in range(17)] for y in range(len(test_patterns))] for z in range(n_subjects)], dtype=float) #needed for 2nd ctrl removal

patterns = np.array([[[0.0 for x in range(17)] for y in range(n_subjects_clean)] for z in range(16)], dtype=float) # steps subjects patterns
patterns_sorted = np.array([[[0.0 for x in range(16)] for y in range(n_subjects_clean)] for z in range(16)], dtype=float) # steps subjects patterns

subjects_clean = np.array([[[0.0 for x in range(17)] for y in range(16)] for z in range(n_subjects_clean)]) # steps patterns subjects
subjects_sorted = np.array([[[0.0 for x in range(16)] for y in range(16)] for z in range(n_subjects_clean)]) # steps patterns subjects in order of test_patterns

steps_by_patt = np.array([[[0.0 for x in range(n_subjects_clean)] for y in range(16)] for z in range(16)], dtype=float) # subjects patterns steps
steps_by_subj = np.array([[[0.0 for x in range(16)] for y in range(n_subjects_clean)] for z in range(16)], dtype=float) # patternvalue at idx z, subjects steps

## Remove second attempt at control patterns ##
for n in range(n_subjects):
    cnt=[0,0,0] #ctrl1, ctrl2, pattcount
    for patt in range(18):
        test_number = int(subjects[n][patt][0])
        if test_number==678:
            if cnt[0]<1:
                subjects_full[n][cnt[2]] = subjects[n][patt]
                cnt[2]+=1
            cnt[0]+=1
        elif test_number==1355:
            if cnt[1]<1:
                subjects_full[n][cnt[2]] = subjects[n][patt]
                cnt[2]+=1
            cnt[1]+=1
        else:
            subjects_full[n][cnt[2]] = subjects[n][patt]
            cnt[2]+=1

## Extract only results from non-outlier subjects ##
idx=0
for n in range(n_subjects):
    if n in subjects_clean_idx:
        subjects_clean[idx] = subjects_full[n]
        # TODO Tap Consistency as well! One function to get clean subjects

        idx += 1

## Sort into ordered by test num and position in the pattern ##
for n in range(n_subjects_clean):
    for i in range(len(subjects_clean[n])):
        test_num = int(subjects_clean[n][i][0])
        for j in range(len(test_patterns)):
            if test_patterns[j]==test_num:
                #print(f"subj {i} patt {j}")
                subjects_sorted[n][j] = subjects_clean[n][i][1:]

                patterns[i][n] = subjects_clean[n][i]
                patterns_sorted[j][n] = subjects_clean[n][i][1:] # patterns sorted into test_patterns order

                for k in range(16): # steps
                    steps_by_patt[k][j][n] = subjects_clean[n][i][k+1] # k+1 to skip the patt num
                    steps_by_subj[k][n][j] = subjects_clean[n][i][k+1]
                    
print(subjects_clean.shape)
print(steps_by_patt.shape)
## Get means and other stats for by pattern ##
pattern_mean = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)
pattern_std = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)
pattern_var = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)
pattern_mean = np.mean(patterns_sorted[:,:,:], axis=1)
pattern_std = np.std(patterns_sorted[:,:,:], axis=1)
pattern_var = np.var(patterns_sorted[:,:,:], axis=1)

## Sort into struct for boxplot plotting ## 
# absolute and true for each
patt_abs = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(16)], dtype=float)
patt_true = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(16)], dtype=float)

subj_true = np.array([[0.0 for x in range(16)] for y in range(n_subjects_clean)], dtype=float)
subj_abs = np.array([[0.0 for x in range(16)] for y in range(n_subjects_clean)], dtype=float)

step_patt_abs = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)
step_patt_true = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)

step_subj_abs = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(16)], dtype=float)
step_subj_true = np.array([[0.0 for x in range(n_subjects_clean)] for y in range(16)], dtype=float)

for p in range(len(test_patterns)):
    for per in range(n_subjects_clean):
        patt_true[p][per] = np.mean(patterns_sorted[p][per]-pattern_mean[p])
        patt_abs[p][per] = np.mean(np.abs(patterns_sorted[p][per]-pattern_mean[p]))

        subj_true[per][p] = np.mean(subjects_sorted[per][p] - pattern_mean[p])
        subj_abs[per][p] = np.mean(np.abs(subjects_sorted[per][p] - pattern_mean[p]))

        for st in range(16):
        ## !! It might be axis=0, not 100% sure
            step_patt_true[st][p] = np.mean(steps_by_patt[st][p]-np.mean(steps_by_patt[:,:,:], axis=0))
            step_patt_abs[st][p] = np.mean(np.abs(steps_by_patt[st][p]-np.mean(steps_by_patt[:,:,:], axis=0)))
            step_subj_true[st][per] = np.mean(steps_by_subj[st][per]-np.mean(steps_by_subj[:,:,:], axis=0))
            step_subj_abs[st][per] = np.mean(np.abs(steps_by_subj[st][per]-np.mean(steps_by_subj[:,:,:], axis=0)))
        
    


# subj

# stepx2


######################################################################################################
############################### Plot data from test experiments. #####################################

## Plotstyles for Consistency ##
## Labels and Titles ##
title_style={
    'fontsize':14,
    'fontfamily':'serif',
    'fontweight':'book'
}
label_style={
    'fontsize':12,
    'fontfamily':'sans-serif',
    'fontweight':'book'
}
## Scatters -----------
scatter_style={
    'color':'dimgrey',
    'marker':'.',
    'linewidth':0.5,
    'alpha':0.6
}
## Lines -----------
dashed_line_style={
    'color':'dimgrey',
    'linestyle':'--',
    'alpha':0.6,
    'linewidth':0.8
}
solid_line_style={
    'color':'dimgrey',
    'linestyle':'-',
    'alpha':0.9,
    'linewidth':0.8
}
## Boxplots -----------
box_style={
    "widths":0.4,
    "patch_artist":True,
    'boxprops': {"alpha":0.3,
                    "facecolor":'lightgrey',
                    "edgecolor":'black'}
}
# - Tap consistency style (Pur-Blu-Gre)




##-----------Plot Subjects-----------##
if _subject:
    print()

##-----------Plot Patterns-----------##
if _pattern:
    fig, (ax, ax1) = plt.subplots(2,1,figsize=(14,8))
    idx=np.arange(16)+1


    # Plot 1 (Pattern v Absolute Error)
    line = ax.axhline(y=0, **solid_line_style)

    # VV Change me VV - currently plots tapping range per pattern, not performance comparison.
    for i in range(len(test_patterns)):
        ax.scatter(np.full(16,i+1), pattern_mean[i], **scatter_style)
    #axbp = ax.boxplot(pattern_mean.T, **box_style)
    ax.set(xticks=idx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.2, 0.8))
    ax.set_title(f"Mean Absolute Difference from Subject Tapped Patterns", **title_style)
    ax.set_xlabel(f"Test Pattern", **label_style)
    ax.set_ylabel(f"Mean Absolute Difference \nfrom Avg. Tap (Velocity)", **label_style)

    # Plot 2 (Pattern vs True Error)
    # scatter
    # boxplot
    ax1.set(xticks=idx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.2, 0.8))
    ax1.set_title(f"True Difference (Mean Error) from Subject Tapped Patterns", **title_style)
    ax1.set_xlabel(f"Test Pattern", **label_style)
    ax1.set_ylabel(f"True Net Difference \nfrom Avg. Tap (Velocity)", **label_style)
    fig.legend()
    plt.show()

##-----------Plot Steps-----------##
if _position:
    fig, (ax, ax1) = plt.subplots(2,1,figsize=(12,8))

    idx=np.arange(16)+1


    # Plot 1 (Pattern v Absolute Error)
    line = ax.axhline(y=0, **dashed_line_style)

    for i in range(16):
        ax.scatter(np.full(16,i+1), step_patt_true[i], **scatter_style)
    axbp = ax.boxplot(step_patt_true.T, **box_style)
    ax.plot(idx, np.mean(step_patt_true[:,:], axis=1), **solid_line_style)
    ax.set(xticks=idx, ylim=(-0.6, 0.8))
    ax.set_title(f"Tap Strength v Pattern Avg \n arranged by step #", **title_style)
    ax.set_xlabel(f"Step #", **label_style)
    ax.set_ylabel(f"Tap Vel. v Pattern Avg.", **label_style)

    # Plot 2 (Pattern vs True Error)
    sidx = np.arange(n_subjects_clean)+1
    line1 = ax1.axhline(y=0, **dashed_line_style)
    for i in range(16):
        ax1.scatter(np.full(n_subjects_clean,i+1), step_subj_true[i], **scatter_style)
    ax1bp = ax1.boxplot(step_subj_true.T, **box_style)
    ax1.plot(idx, np.mean(step_subj_true[:,:], axis=1), **solid_line_style)
    ax1.set(xticks=idx, ylim=(-0.6, 0.8))
    ax1.set_title(f"Tap Strength v Subj Avg \n arranged by step #", **title_style)
    ax1.set_xlabel(f"Step #", **label_style)
    ax1.set_ylabel(f"Tap Vel. v Subj Avg.", **label_style)
    fig.legend()
    plt.show()

    print()

##-----------Plot Tap Consistency-----------##
if _tapcalibration:
    print()


# TODO: (true and abs errors necessary to highlight different aspects of behavior)
# - Error by subject over all patterns (non-outlier subject trend of behavior)
# - Error by pattern over all subjects (were any patterns problematic)
# - Error by step in pattern (did people exhibit behavior related to metrical position)
# - Error by order of pattern presentation (total pattern error v pattern position in test order)


# - Check distribution of mean errors for subjects and alg predictions (one large error can skew mean)
# - Attempt scaling subjects a. overall and b. by pattern inidividually (normalize by individual or by tap tap results)