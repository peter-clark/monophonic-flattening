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
n_subjects=40
_wholedataset = False

# Which analysis
_coordinates = False
_control = True
_subject = False
_pattern = True
_position = False
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

######################################################################################################
################################## Outlier Selection #################################################
## Variable Declaration ##
control_precision = np.asarray([[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)], dtype=float)
control_accuracy = np.asarray([[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)], dtype=float)
control_accuracy_abs = np.asarray([[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)], dtype=float)
control_means = np.asarray([[0.0 for x in range(16)] for y in range(2)], dtype=float)

## Means for Accuracy Comparison ##
for n in range(n_subjects):
    control_means[0] += (controls[n][0] + controls[n][1])/2
    control_means[1] += (controls[n][2] + controls[n][3])/2
control_means[:] /= n_subjects
print(control_means)
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
print(f"[M={np.mean(np.abs(precision_means[:,0]))}, SD={np.std(precision_means[:,0]):.3f}]")
print(f"[V={np.var(precision_means[:,0]):.3f}, CV={(np.std(precision_means[:,0])/np.mean(np.abs(precision_means[:,0]))):.3f}]")
print("1355:")
print(f"[M={np.mean(np.abs(precision_means[:,1]))}, SD={np.std(precision_means[:,1]):.3f}]")
print(f"[V={np.var(precision_means[:,1]):.3f}, CV={(np.std(precision_means[:,1])/np.mean(np.abs(precision_means[:,1]))):.3f}]")

## Accuracy Stats ##
ctrl_means = np.mean(control_accuracy, axis=2)
ctrl_means = np.asarray(ctrl_means, dtype=float)
print("\nACCURACY:\n678:")
print(f"[M={np.mean(np.abs(ctrl_means[:,0]))}, SD={np.std(ctrl_means[:,0]):.3f}]")
print(f"[V={np.var(ctrl_means[:,0]):.3f}, CV={(np.std(ctrl_means[:,0])/np.mean(np.abs(ctrl_means[:,0]))):.3f}]")
print("1355:")
print(f"[M={np.mean(np.abs(ctrl_means[:,1]))}, SD={np.std(ctrl_means[:,1]):.3f}]")
print(f"[V={np.var(ctrl_means[:,1]):.3f}, CV={(np.std(ctrl_means[:,1])/np.mean(np.abs(ctrl_means[:,1]))):.3f}]")

ctrl_means_abs = np.mean(control_accuracy_abs, axis=2)
ctrl_means_abs = np.asarray(ctrl_means_abs, dtype=float)
print("\nABS ACCURACY:\n678:")
print(f"[M={np.mean(np.abs(ctrl_means_abs[:,0]))}, SD={np.std(ctrl_means_abs[:,0]):.3f}]")
print(f"[V={np.var(ctrl_means_abs[:,0]):.3f}, CV={(np.std(ctrl_means_abs[:,0])/np.mean(np.abs(ctrl_means_abs[:,0]))):.3f}]")
print("1355:")
print(f"[M={np.mean(np.abs(ctrl_means_abs[:,1]))}, SD={np.std(ctrl_means_abs[:,1]):.3f}]")
print(f"[V={np.var(ctrl_means_abs[:,1]):.3f}, CV={(np.std(ctrl_means_abs[:,1])/np.mean(np.abs(ctrl_means_abs[:,1]))):.3f}]")

## Plot Precision and Accuracy ## ------------------------------------------------------------------------------
if _control:
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(18,7), gridspec_kw={'width_ratios': [3,3,3]}) 

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    ax3.set_aspect('equal')

    ## Precision by Control Pattern ##
    p1=(np.std(precision_means[:,0])*2,-np.std(precision_means[:,1])*2)
    p2=(-np.std(precision_means[:,0])*2,np.std(precision_means[:,1])*2)

    ax1.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax1.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax1.add_patch(rect)
    ax1.scatter(precision_means[:,0], precision_means[:,1], marker='x', color='lightgreen', label="Intra-Pattern Precision")
    for n in range(n_subjects): # Subject # Labels
        ax1.text(precision_means[n,0], precision_means[n,1], str(n+1), size='x-small')
    ax1.set(xlim=[-0.5,0.5], ylim=[-0.5,0.5], xticks=np.arange(-0.5,0.5,0.1), yticks=np.arange(-0.5,0.5,0.1))
    ax1.grid(color='lightgrey', linewidth=0.3, alpha=0.4)

    ## Accuracy by Control Pattern ##
    p1=(np.std(ctrl_means[:,0])*2,-np.std(ctrl_means[:,1])*2)
    p2=(-np.std(ctrl_means[:,0])*2,np.std(ctrl_means[:,1])*2)
    rect = mpl.patches.Rectangle((p2[0], p1[1]), p1[0] - p2[0], p2[1] - p1[1], linewidth=0.8, edgecolor='green', facecolor='none', linestyle='--')
    ax2.add_patch(rect)
    ax2.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax2.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax2.scatter(ctrl_means[:,0], ctrl_means[:,1], marker='.', color='lightcoral', label="Sum Mean Accuracy")
    for n in range(n_subjects): # Subject # Labels
        ax2.text(ctrl_means[n,0], ctrl_means[n,1], str(n+1), size='x-small')
    ax2.set(xlim=[-1,1], ylim=[-1,1], xticks=np.arange(-1,1,0.25), yticks=np.arange(-1,1,0.25))
    ax2.grid(color='lightgrey', linewidth=0.3, alpha=0.4)

    ## Absolute Accuracy by Control Pattern ##
    p3=(np.mean(ctrl_means_abs[:,0])+(np.std(ctrl_means_abs[:,0])*2), np.mean(ctrl_means_abs[:,1])-(np.std(ctrl_means_abs[:,1])*2))
    p4=((np.mean(ctrl_means_abs[:,0])-(np.std(ctrl_means_abs[:,0]))*2), np.mean(ctrl_means_abs[:,1])+(np.std(ctrl_means_abs[:,1])*2))
    rect = mpl.patches.Rectangle((p4[0], p3[1]), p3[0] - p4[0], p4[1] - p3[1], linewidth=0.8, edgecolor='blue', facecolor='none', linestyle='--')
    ax3.add_patch(rect)
    ax3.axhline(y=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax3.axvline(x=0.0, linewidth=0.8, linestyle="--", color='grey', alpha=0.8)
    ax3.scatter(ctrl_means_abs[:,0], ctrl_means_abs[:,1], marker='.', color='lightblue', label="Sum Abs. Mean Accuracy")
    for n in range(n_subjects):
        ax3.text(ctrl_means_abs[n,0], ctrl_means_abs[n,1], str(n+1), size='x-small')
    ax3.set(xlim=[0,1], ylim=[0,1], xticks=np.arange(0,1,0.25), yticks=np.arange(0,1,0.25))
    ax3.grid(color='lightgrey', linewidth=0.3, alpha=0.4)

    
    fig.legend()
    plt.show()

######################################################################################################
########################## Sort cleaned subjects, patterns and steps #################################
## Variable Declaration ##
patterns = np.array([], dtype=float)
steps = np.array([], dtype=float) # step in pattern