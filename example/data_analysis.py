import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
tf.make_pos_grid(mds_pos,sections,pattern_idxs,all_names,names_in_section, True) # bool is show plot
test_patterns = [894, 423, 1367, 249, 939, 427, 590, 143, 912, 678, 1355, 580, 1043, 673, 1359, 736]

# Load and read CSV
with open(tap_file) as results: 
    reader = csv.reader(results)
    i=0
    for row in reader:
        data.append(row)
    results.close()

# Define last two algs
sc1 = [0.0 for x in range(16)]
sc2 = [0.0 for x in range(16)]
_idx = [int(x) for x in range(16)]
idx = np.array(_idx)
idx = idx + 1
data_anova=[[0.0 for x in range(len(data)-1)] for y in range(6)]
# Calculate 5th 6th alg
for test in range(len(data)):
    if test==0:
        test+=1
    for i in range(16):
        sc1[i]= float(data[test][20+i])*float(data[test][36+i])
        sc2[i]= float(data[test][52+i])*float(data[test][68+i])
        data[test].append(sc1[i])
    for i in range(16):
        data[test].append(sc2[i])

# ------------------------------------------------------------ #
# Sort through data
counts = np.array([0 for x in range(16)], dtype=float)
avgs = np.array([[0.0 for x in range(16)] for y in range(16)], dtype=float)
algs = np.array([[[0.0 for x in range(16)] for y in range(6)]for z in range(16)], dtype=float)
alg_names = ["cont1","disc1","cont2","disc2","semicont1","semicont2"]
for i in range(len(data)):
    if i!=0: # skip first row
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
                avgs[y] += tap
                counts[y] += 1
                algs[y][0] = c_one
                algs[y][1] = d_one
                algs[y][2] = c_two
                algs[y][3] = d_two
                algs[y][4] = sc_one
                algs[y][5] = sc_two
                #print(algs)

avgs = avgs / counts

# Remove mistaps (if needed)
""" for i in range(len(avgs)):
    _avg=[0.0 for x in range(len(avgs))]
    replace = _avg
    _avg = np.where(avgs[i] > np.mean(avgs[i]), avgs[i], replace)
    avgs[i]=_avg
    #print(avgs[i])
 """
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
# Print error-test results
for tn in range(len(test_names)): # iterate through test types
    print(f"{test_names[tn]}:-------------:")
    for algtype in range(len(alg_names)): # iterate through 6 alg types
        print(f"{alg_scores[tn][algtype]:1.4f} <- {alg_names[algtype]}")

# ANOVA
f_stat,p_val = stats.f_oneway(data_anova[0],data_anova[1],data_anova[2],data_anova[3],data_anova[4],data_anova[5])
print(f"{test_patterns[i]} ANOVA: ")
print(f"F-Statistics: {f_stat:1.4f}")
print(f"P-Values: {p_val:1.6f}")

# Tukey's HSD
_tukey = True
if _tukey:
    # Reorder data for tukey
    pre_df=[]
    for i in range(6):
        print(f"len d_anova {len(data_anova[i])}")
        for j in range(len(data_anova[i])):
            pre_df.append([i, data_anova[i][j]])

    # Do Tukey's HSD
    alg_val=[row[0] for row in pre_df]
    dist_val = [row[1] for row in pre_df]
    df = pd.DataFrame({'algorithm': alg_val, 'distance':dist_val})
    tukey = pairwise_tukeyhsd(endog=df["distance"],groups=df['algorithm'], alpha=0.05)
    np.set_printoptions(precision=6)
    print(tukey)
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    # Set the desired precision for p-adj values
    pd.set_option('display.float_format', '{:.12f}'.format)

    # Print the DataFrame
    print(tukey_df)

    #seaborn.boxplot(x=alg_val, y=(dist_val), color="lightcoral")
    #plt.show()


# Plot stuff
""" 
idx2=idx-1
plt.plot(idx2, c_one, color='darkred', marker='o', linestyle='--', label="Predicted Value")
plt.ylabel("ABS Distance in Velocity Value")
plt.xlabel("Alg [Cont, Disc, Semi-Cont]")
plt.title("Distances between Alg and Tap")
plt.ylim([0,1])
plt.show()
 """


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

# Plot
if _by_pattern:
    _firstplot=False
    _secondplot=True
    if _firstplot:
        for i in range(16):
            plt.errorbar(idx,patt_means[i],yerr=patt_stds[i], color='grey', linewidth=1)
            plt.plot(idx,patt_means[i], marker="o", color='lightcoral', linestyle='-', label="Avg. Tap")
            plt.plot(idx, algs[i][2], marker='x', linestyle='--',color='deepskyblue', label='Cont.2 (DSM)')
            #plt.plot(idx, algs[i][0], marker='x', linestyle='--', color='mediumpurple',label='Cont.1 (DS)')
            plt.title(f"#{test_patterns[i]} - {all_names[int(test_patterns[i])]}\n vs. Algorithm Predictions")
            plt.ylabel("Normalized Velocity")
            plt.xlabel("Step in Pattern")
            plt.legend()
            #plt.plot(idx,patt_stds)
            plt.show()
    
    if _secondplot:
        alg = 2                 #  <---- pick flattening alg here
        
        for j in range(4):
            i=j*4
            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            p1=axes[0,0]
            p2=axes[0,1]
            p3=axes[1,0]
            p4=axes[1,1]
            #topleft
            p1.errorbar(idx, patt_means[i],yerr=patt_stds[i], color='grey', linewidth=1)
            p1.plot(idx,patt_means[i], marker="o", color='lightcoral', linestyle='-', label="Avg. Tap")
            line1,=p1.plot(idx,patt_means[i], marker="o", color='lightcoral', linestyle='-', label="Avg. Tap")
            p1.plot(idx, algs[i][alg], marker='x', linestyle='--',color='deepskyblue', label='Cont.2 (DSM)')
            line2,=p1.plot(idx, algs[i][alg], marker='x', linestyle='--',color='deepskyblue', label='Cont.2 (DSM)')
            p1.set_title(f"#{test_patterns[i]} - {all_names[int(test_patterns[i])]}")
            p1.set_ylabel("Normalized Velocity")
            p1.set_xlabel("Step in Pattern")
            
            #topright
            p2.errorbar(idx, patt_means[i+1],yerr=patt_stds[i+1], color='grey', linewidth=1)
            p2.plot(idx,patt_means[i+1], marker="o", color='lightcoral', linestyle='-', label="Avg. Tap")
            p2.plot(idx, algs[i+1][alg], marker='x', linestyle='--',color='deepskyblue', label='Cont.2 (DSM)')
            p2.set_title(f"#{test_patterns[i+1]} - {all_names[int(test_patterns[i+1])]}")
            p2.set_ylabel("Normalized Velocity")
            p2.set_xlabel("Step in Pattern")

            #bottom left
            p3.errorbar(idx, patt_means[i+2],yerr=patt_stds[i+2], color='grey', linewidth=1)
            p3.plot(idx,patt_means[i+2], marker="o", color='lightcoral', linestyle='-', label="Avg. Tap")
            p3.plot(idx, algs[i+2][alg], marker='x', linestyle='--',color='deepskyblue', label='Cont.2 (DSM)')
            p3.set_title(f"#{test_patterns[i+2]} - {all_names[int(test_patterns[i+2])]}")
            p3.set_ylabel("Normalized Velocity")
            p3.set_xlabel("Step in Pattern")

            #bottom right
            p4.errorbar(idx, patt_means[i+3],yerr=patt_stds[i+3], color='grey', linewidth=1)
            p4.plot(idx,patt_means[i+3], marker="o", color='lightcoral', linestyle='-', label="Avg. Tap")
            p4.plot(idx, algs[i+3][alg], marker='x', linestyle='--',color='deepskyblue', label='Cont.2 (DSM)')
            p4.set_title(f"#{test_patterns[i+3]} - {all_names[int(test_patterns[i+3])]}")
            p4.set_ylabel("Normalized Velocity")
            p4.set_xlabel("Step in Pattern")

            plt.suptitle("Continuous2 (DSM) vs Average Tapped Pattern")
            fig.legend(handles=[line1,line2], loc='upper left', labels=['Avg. Tap','Cont.2 (DSM)'])
            fig.tight_layout()
            plt.show()