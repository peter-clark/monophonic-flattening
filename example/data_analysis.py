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
idx = np.array(_idx, dtype=int)
idx = idx + 1
data_anova=[[0.0 for x in range(len(data)-1)] for y in range(6)]
n_subjects=20
by_person =[[[0.0 for x in range(17)] for y in range(18)] for z in range(n_subjects)] # [patt# tapresults]x18tests
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
t_count=0
p_count=0
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
                #print(f"{y}-{test_patterns[y]}")
                avgs[y] += tap
                counts[y] += 1
                algs[y][0] = c_one
                algs[y][1] = d_one
                algs[y][2] = c_two
                algs[y][3] = d_two
                algs[y][4] = sc_one
                algs[y][5] = sc_two
                #print(algs)
        
        # find by person
        line = [0.0 for x in range(17)]
        line[0]=int(data[i][2])
        line[1:16]=tap
        if (i-1)%18==0 and (i-1)!=0:
            p_count += 1
            t_count = 0
        if(p_count!=n_subjects):
            by_person[p_count][t_count] = np.asarray(line, dtype=float)
            #print(f"P:{p_count} t:{t_count}")
            t_count +=1
#print(by_person)
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
    _secondplot=False
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


# By Person Analysis
person_control_mean_diff = [[[0.0 for x in range(16)] for y in range(4)] for z in range(n_subjects)]
pcmd = [[[0.0 for x in range(16)] for y in range(2)] for z in range(n_subjects)]

_byperson = True
_control = False
_all = False
_subjectaverageerror = True
_patternaverageerror = True
if _byperson:
    if _control:
        control1=678
        control2=1355
        # [[17]x18]
        for person in range(len(by_person)): # per subject:
            cnt=0
            cnt2=2
            for test in range(len(by_person[person])): # per test
                if by_person[person][test][0] == control1:
                    for i in range(16):
                        person_control_mean_diff[person][cnt][i] = by_person[person][test][i+1] # skip patt# in first cell
                    cnt+=1
                elif by_person[person][test][0] == control2:
                    for i in range(16):
                        person_control_mean_diff[person][cnt2][i] = by_person[person][test][i+1] # skip patt# in first cell
                    cnt2+=1
            for i in range(16):
                pcmd[person][0][i]=(person_control_mean_diff[person][0][i]-person_control_mean_diff[person][1][i])
                pcmd[person][1][i]=(person_control_mean_diff[person][2][i]-person_control_mean_diff[person][3][i])
                # abs
                # pcmd[person][0][i]=np.abs(person_control_mean_diff[person][0][i]-person_control_mean_diff[person][1][i])
                # pcmd[person][1][i]=np.abs(person_control_mean_diff[person][2][i]-person_control_mean_diff[person][3][i])

        ctrl1 = np.array([0.0 for x in range(16)],dtype=float)
        ctrl2 = np.array([0.0 for x in range(16)],dtype=float)
        _ctrl1 = np.array([[0.0 for x in range(16)]for y in range(n_subjects)],dtype=float)
        _ctrl2 = np.array([[0.0 for x in range(16)] for y in range(n_subjects)],dtype=float)

        fig2, (plt1,plt2) = plt.subplots(2, 1, figsize=(11,8))
        one_box=np.array([[0.0 for y in range(16)] for x in range(n_subjects)],dtype=float)
        two_box=np.array([[0.0 for y in range(16)] for x in range(n_subjects)],dtype=float)
        for i in range(n_subjects):
            plt1.plot(idx, pcmd[i][0], marker="x", color='lightcoral', linestyle='--', label="Patt. 678",alpha=.65)
            plt2.plot(idx, pcmd[i][1], marker="x", color='lightcoral', linestyle='--', label="Patt. 1355",alpha=.65)
            ctrl1 += pcmd[i][0]
            one_box[i]=pcmd[i][0]
            _ctrl1[i] = pcmd[i][0]
            ctrl2 += pcmd[i][1]
            two_box[i]=pcmd[i][1]
            _ctrl2[i] = pcmd[i][1]
        #print(len(one_box))
        plt1.boxplot(one_box[:-2])
        plt1.errorbar(idx,ctrl1/n_subjects, yerr=np.std(_ctrl1, axis=0), color='black', linewidth=1)
        plt1.plot(idx,ctrl1/n_subjects, marker='o', linestyle='-', color='black', label='Mean(678)')
        plt1.set_xlabel("Step in Pattern")
        plt1.set_ylabel("Difference in Velocity")

        plt2.boxplot(two_box[:-2])
        plt2.errorbar(idx,ctrl2/n_subjects, yerr=np.std(_ctrl2, axis=0), color='black', linewidth=1)
        plt2.plot(idx,ctrl2/n_subjects, marker='o', linestyle='-',color='black', label='Mean(1355)')
        plt2.set_xlabel("Step in Pattern")
        plt2.set_ylabel("Difference in Velocity")
        plt.suptitle("Mean Tapped Difference in Control Patterns")

        fig2.tight_layout()
        plt.show()
    
    colormap = mpl.colormaps['winter'].resampled(n_subjects)
    def custom_formatter(x,pos):
        return test_patterns[str(x)]
    if _all:
        for person in range(len(by_person)): # per subject:
            mean_diff=np.array([[0.0 for x in range(16)] for y in range(18)])
            stds=np.array([[0.0 for x in range(16)] for y in range(18)])
            #[894, 423, 1367, 249, 939, 427, 590, 143, 912, 678, 1355, 580, 1043, 673, 1359, 736]
            # go through subjects tests
            for test in range(len(by_person[person])):
                for patt in range(len(test_patterns)): # check which pattern it is
                    if by_person[person][test][0]==test_patterns[patt]:
                        #print(f"{test_patterns[patt]} - {all_names[test_patterns[patt]]}")
                        #print(len(by_person[person][test]))
                        mean_diff[test] =  by_person[person][test][1:-1] - patt_means[patt]
                        #print(by_person[person][test][1:])
                #print(f"{by_person[1][test]}")        #print(mean_diff[test])


            # get overall mean difference from patt (check for means in patt_means[], test_patterns)
            # add to means, get std over mean diff
            lines=[]
            figure = plt.figure(figsize=(12,6))
            ax = figure.add_subplot()
            #ax.boxplot(mean_diff)
            for i in range(n_subjects):
                a = min(1, pow(1-np.mean(mean_diff[i]),2) ) # set alpha
                ax.plot(idx, np.mean(mean_diff, axis=0), color='red', linestyle='--', marker='o')
                ax.errorbar(idx, np.mean(mean_diff, axis=0),yerr=np.std(mean_diff, axis=0), linewidth=1, color='black', capsize=2)
                ax.scatter(idx, mean_diff[i], label=f"{by_person[person][test][0]}", color='lightcoral',linestyle='-', marker='x', alpha=(a))

            ax.set(xticks=idx, xticklabels=[str(x) for x in test_patterns])
            ax.set_title(f"Subject {person}")
            ax.set_xlabel("Test Pattern")
            ax.set_ylabel("Difference from Average")
            ax.axhline(y=0,color='black', alpha=0.8)
            plt.show()
            #colormap(i)

            if person==2: # breakpoint for how many subjects to see
                break;
    
    
    if _subjectaverageerror:
        # Subject average error (over all patterns) x subject
        control_patterns = np.array([[[0.0 for x in range(16)] for y in range(4)] for z in range(n_subjects)],dtype=float)
        control_differences = np.array([[0.0 for y in range(2)] for z in range(n_subjects)], dtype=float)
        sae = np.array([0.0 for x in range(len(by_person))])
        sae_box = np.array([[0.0 for y in range(18)] for z in range(len(by_person))])
        control1=678
        c1_idx = 0
        control2=1355
        c2_idx = 2
        for person in range(len(by_person)):
            mean_diff=np.array([[0.0 for x in range(16)] for y in range(18)])
            stds=np.array([[0.0 for x in range(16)] for y in range(18)])

            for test in range(len(by_person[person])):
                
                # get answers for two control patterns
                if by_person[person][test][0] == control1:
                    control_patterns[person][c1_idx]=by_person[person][test][1:-1]
                    c1_idx+=1
                if by_person[person][test][0] == control2:
                    control_patterns[person][c2_idx]=by_person[person][test][1:-1]
                    c2_idx+=1
                
                # check which pattern it is
                for patt in range(len(test_patterns)): 
                    if by_person[person][test][0]==test_patterns[patt]:
                        mean_diff[test] =  by_person[person][test][1:-1] - patt_means[patt]
                
                sae_box[person][test] = np.mean(mean_diff[test])
                sae[person] = np.mean(sae_box[person][test])
            
            control_differences[person][0]=np.mean(control_patterns[person][0]-control_patterns[person][1])
            control_differences[person][1]=np.mean(control_patterns[person][2]-control_patterns[person][3])
            c1_idx=0
            c2_idx=2
            
            
        # Plot of MAE  of subjects
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot()
        ax.axhline(y=0,color='black', alpha=0.8, linewidth=1,label='No Error')
        s_idx = np.array(np.arange(n_subjects))
        #s_idx=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],dtype=int)
        ax.set(xticks=s_idx+2, xticklabels=[str(x) for x in s_idx], ylim=(-0.6,0.6))
        for i in range(len(by_person)):
            ax.scatter(np.full(len(sae_box[i]),i+1),sae_box[i], color='lightcoral', linewidth=0.5, marker='x', alpha=0.6)
        ax.scatter(s_idx+1,control_differences[:,0],color="purple",marker='p', label='Subject Control Error (678)')
        ax.scatter(s_idx+1,control_differences[:,1],color="teal",marker='p', label='Subject Control Error (1355)')
        ax.boxplot(sae_box.T)
        ax.scatter(s_idx+1, sae, color='lightcoral', linestyle='-', marker='o', label="Subject Avg. Err.")
        ax.scatter
        ax.set_title(f"Subject v. Subject Average Error (all patts)")
        ax.set_xlabel("Subject #")
        ax.set_ylabel("Average Velocity Error from Overall Mean Tapped Pattern")
        plt.legend()
        plt.show()


        # Box plot of MAE of both control patterns
        fig2 = plt.figure(figsize=(12,6))
        axx = fig2.add_subplot()
        #print(f"{len(avgs[9])}-{avgs[9]}")
        #print(f"{len(control_patterns[:,0])}-{control_patterns[:,0]}")
        
        ax.scatter(s_idx+1,(control_differences[:,0]-avgs[9]),color="purple",marker='p', label='Subject Control Error (678)')

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
                        mean_diff[test] =  by_person[person][test][1:-1] - patt_means[patt]
                    
                        pae_box[patt][person] = np.mean(mean_diff[test])
        
        for patt in range(len(test_patterns)):
            pae[patt]=np.mean(pae_box[patt])
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot()
        pidx=np.arange(16)+1

        ax.axhline(y=0,color='black', alpha=0.6)
        for i in range(len(test_patterns)):
            ax.scatter(np.full(len(pae_box[i]),i+1),pae_box[i], color='lightcoral', linewidth=0.5, marker='x', alpha=0.6)
        ax.boxplot(pae_box.T)
        ax.scatter(pidx, pae,color='lightcoral', linestyle='-', marker='o', label="Pattern Avg. Err.")
        
        ax.set(xticks=pidx, xticklabels=[str(x) for x in test_patterns], ylim=(-0.6,0.6))
        ax.set_title(f"Average Velocity Error from Mean by Pattern")
        ax.set_xlabel("Test Pattern")
        ax.set_ylabel("Difference from Mean (tap velocity)")
        plt.show()