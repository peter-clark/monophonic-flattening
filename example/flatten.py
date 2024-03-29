import os
import numpy as np
import descriptors as desc
import re

def parse(line):
    line = str(line)
    regex = r"[-+]?\d*\.\d+|\d" # searches for all floats or integers
    list = re.findall(regex, line)
    output = [float(x) for x in list]
    return output

def is_Max(current_max_num, num):
    return num if(num>current_max_num)else current_max_num

## Normalize a 4x16-step pattern of velocity values
def normalize_velocity(patterns, max_in_pattern):
    for i in range(2):
        for j in range(len(patterns[i])):
                patterns[i][j] = patterns[i][j]/max_in_pattern[0]
                patterns[i+2][j] = patterns[i+2][j]/max_in_pattern[1]
    means = [np.sum(patterns[0])/len(patterns[0]), np.sum(patterns[2])/len(patterns[2])]
    return patterns, means

## Finds approriate frequency channel for midi note
def find_LMH(note):
    if(int(note)==0):
        return []
    channel = desc.GM_dict[int(note)][1]
    if(channel=="low"):
        n = 1
    elif(channel=="mid"):
        n = 2
    else:
        n = 3 # "high"
    return n

## Used in flat_from_patt
def get_LMH(pattern):
    pattern_LMH = []
    for step in range(len(pattern)):
        lmh = []
        for note in pattern[step]:
            if pattern[step] != "":
                lmh.append(find_LMH(note))
        pattern_LMH.append(lmh)
    return pattern_LMH


## Input: pattern of 8-instrument midi notes in array of 16 steps
## Output: four flattened representations in array (2 continous, 2 discrete)
def flat_from_patt(pattern):
    # Initialize variables
    pattern_LMH = get_LMH(pattern) # LOW MID HIGH
    pattern_LMH_count=[[0 for x in range(len(pattern_LMH))] for y in range(3)]
    total_count = [0.0 for x in range(4)]
    flattened_patterns = [[0.0 for x in range(len(pattern_LMH))]for y in range(6)]
    true_sync_salience = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1]
    metric_sal_strength = [4,0,1,0, 2,0,1,0, 3,0,1,0, 2,0,1,0]
    sync_strength = [0,1,0,2, 0,1,0,3, 0,1,0,2, 0,1,0,4]
    sync_strength = [0,4,0,1, 0,2,0,1, 0,3,0,1, 0,2,0,1]

    # Count multi-hits in same channel on step
    for i in range(len(pattern_LMH)):
        for j in range(len(pattern_LMH[i])):
            pattern_LMH_count[0][i] += 1 if pattern_LMH[i][j]==1 else 0 # LOW
            pattern_LMH_count[1][i] += 1 if pattern_LMH[i][j]==2 else 0 # MID
            pattern_LMH_count[2][i] += 1 if pattern_LMH[i][j]==3 else 0 # HIGH
    for i in range(3): # GET TOTAL COUNT
        total_count[i] = float(np.sum(pattern_LMH_count[i]))
        total_count[3] += float(np.sum(pattern_LMH_count[i]))

    # Initialize variables for flattening
    density = [0.0 for x in range(3)]
    salience = [0.0 for x in range(4)]
    norm_salience = [0.0 for x in range(3)]
    means = [0.0,0.0]
    maxes = [0.0,0.0]
    for i in range(3):
        density[i] = 0.0 if total_count[3]==0 else float(total_count[i]/total_count[3])
        salience[i] = 0.0 if density[i]<=0.0 else float(1/density[i])
        salience[3] += salience[i]
    for i in range(3):
        norm_salience[i] = salience[i]/salience[3]


    # Testing other onset density method
    norm_salience = [3.0,2.0,1.0]
    #tmp = np.array([3.0,2.0,1.0], dtype=float)
    #for i in range(3):
    #    norm_salience[i] *= tmp

    # Loop through pattern
    for i in range(len(pattern_LMH)):
        if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
            note_values = [norm_salience[0]*pattern_LMH_count[0][i],norm_salience[1]*pattern_LMH_count[1][i],norm_salience[2]*pattern_LMH_count[2][i]]
            note_values_density_sync = [note_values[0],note_values[1],note_values[2]]
            note_values_density_sync_meter = [note_values[0],note_values[1],note_values[2]]

            ## FLATTENING ALGORITHMS
            # [1] Normalized Density Salience and Syncopation Strength
            if i>0:
                if(i<len(pattern_LMH)):
                    if(true_sync_salience[i-1]>true_sync_salience[i]): # if note is syncop
                        if(pattern_LMH_count[0][i-1]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][i-1]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][i-1]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])
            """ if(i<len(pattern_LMH)-1):
                if(true_sync_salience[i]>true_sync_salience[i+1]): # if note is syncop
                    if(pattern_LMH_count[0][i+1]==0):
                        note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                    if(pattern_LMH_count[1][i+1]==0):
                        note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                    if(pattern_LMH_count[2][i+1]==0):
                        note_values_density_sync[2] += (note_values[2]*sync_strength[i])
            if(i==len(pattern_LMH)-1):
                    if(pattern_LMH_count[0][0]==0):
                        note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                    if(pattern_LMH_count[1][0]==0):
                        note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                    if(pattern_LMH_count[2][0]==0):
                        note_values_density_sync[2] += (note_values[2]*sync_strength[i]) """
            flattened_patterns[0][i] = np.sum(note_values_density_sync)
            flattened_patterns[1][i] = np.sum(note_values_density_sync)
            means[0] += np.sum(note_values_density_sync)
            maxes[0] = is_Max(maxes[0], np.sum(note_values_density_sync))

            # [2] Normalized Density Salience, Metric Salience, Syncopation Strength
            if(i<len(pattern_LMH)-1):
                if(metric_sal_strength[i]>metric_sal_strength[i+1]): # if meter is reinforced
                    if(pattern_LMH_count[0][i+1]==0):
                        note_values_density_sync_meter[0] += (note_values[0]*metric_sal_strength[i])
                    if(pattern_LMH_count[1][i+1]==0):
                        note_values_density_sync_meter[1] += (note_values[1]*metric_sal_strength[i])
                    if(pattern_LMH_count[2][i+1]==0):
                        note_values_density_sync_meter[2] += (note_values[2]*metric_sal_strength[i])

            note_values_density_sync_meter[0] += (note_values_density_sync[0])
            note_values_density_sync_meter[1] += (note_values_density_sync[1])
            note_values_density_sync_meter[2] += (note_values_density_sync[2])
            flattened_patterns[2][i] = np.sum(note_values_density_sync_meter)
            flattened_patterns[3][i] = np.sum(note_values_density_sync_meter)
            means[1] += np.sum(note_values_density_sync_meter)
            maxes[1] = is_Max(maxes[1], np.sum(note_values_density_sync_meter))
    flattened_patterns, means = normalize_velocity(flattened_patterns, maxes)

    # Convert to boolean/discrete once for each algorithm
    for step in range(len(pattern_LMH)):
        flattened_patterns[1][step]=1 if flattened_patterns[1][step]>=means[0] else 0
        flattened_patterns[3][step]=1 if flattened_patterns[3][step]>=means[1] else 0
        flattened_patterns[4][step]=flattened_patterns[0][step] if flattened_patterns[1][step]==1 else 0
        flattened_patterns[5][step]=flattened_patterns[2][step] if flattened_patterns[3][step]==1 else 0

    return flattened_patterns

def get_lmh_counts(pattern):
    pattern_LMH = get_LMH(pattern) # LOW MID HIGH
    pattern_LMH_count=[[0 for x in range(len(pattern_LMH))] for y in range(3)]
     # Count multi-hits in same channel on step
    for i in range(len(pattern_LMH)):
        for j in range(len(pattern_LMH[i])):
            pattern_LMH_count[0][i] += 1 if pattern_LMH[i][j]==1 else 0 # LOW
            pattern_LMH_count[1][i] += 1 if pattern_LMH[i][j]==2 else 0 # MID
            pattern_LMH_count[2][i] += 1 if pattern_LMH[i][j]==3 else 0 # HIGH
    return pattern_LMH_count


def onset_density(pattern):
    arr = np.array([0.0 for x in range(16)], dtype=float)

    pattern_LMH_count = get_lmh_counts(pattern)
    pattern_LMH_count = np.array(pattern_LMH_count, dtype=float)

    arr = np.sum(pattern_LMH_count, axis=0) / np.max(np.sum(pattern_LMH_count, axis=0)) # Normalize
    return arr

def witek_scaling(pattern):
    arr = np.array([0.0 for x in range(16)], dtype=float)
    
    pattern_LMH_count = get_lmh_counts(pattern)
    pattern_LMH_count = np.array(pattern_LMH_count, dtype=float)
    
    # scale by 3/2/1
    pattern_LMH_count[0] *= 3 # low
    pattern_LMH_count[1] *= 2 # mid

    arr = np.sum(pattern_LMH_count, axis=0) / np.max(np.sum(pattern_LMH_count, axis=0)) # Normalize
    #print(arr.shape)
    return arr

## These were forward syncopations (preceding the high meter rest)
## They are currently backwards syncopations (after) and perform better than the forwards ones.
def syncopation(pattern, type):
    arr = np.array([0.0 for x in range(16)], dtype=float)
    pattern_LMH_count = get_lmh_counts(pattern)
    pattern_LMH_count = np.array(pattern_LMH_count, dtype=float)

    true_sync_salience = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1]
    sync_strength = [0,1,0,2, 0,1,0,3, 0,1,0,2, 0,1,0,4]
    sync_strength = [0,4,0,1, 0,2,0,1, 0,3,0,1, 0,2,0,1]


    for i in range(16):
        # Code from sync part of flattening algorithm above
        if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
            note_values = [pattern_LMH_count[0][i], pattern_LMH_count[1][i], pattern_LMH_count[2][i]]
            for note in range(len(note_values)): # this loop removes multi hits
                if note_values[note]>=1:
                    note_values[note] = np.min([note_values[note], 1])
            note_values_density_sync = [note_values[0],note_values[1],note_values[2]]

            if i>0:
                if(i<len(pattern_LMH_count)):
                    if(true_sync_salience[i-1]>true_sync_salience[i]): # if note is syncop
                        if(pattern_LMH_count[0][i-1]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][i-1]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][i-1]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])
            """ if(i<15):
                if(true_sync_salience[i]>true_sync_salience[i+1]): # if note is syncop
                    if(pattern_LMH_count[0][i+1]==0):
                        note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                    if(pattern_LMH_count[1][i+1]==0):
                        note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                    if(pattern_LMH_count[2][i+1]==0):
                        note_values_density_sync[2] += (note_values[2]*sync_strength[i])
            if(i==15):
                    if(pattern_LMH_count[0][0]==0):
                        note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                    if(pattern_LMH_count[1][0]==0):
                        note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                    if(pattern_LMH_count[2][0]==0):
                        note_values_density_sync[2] += (note_values[2]*sync_strength[i]) """
            if type==2:
                note_values_density_sync[0] *= 3 # low
                note_values_density_sync[1] *= 2 # mid
            arr[i] = np.sum(note_values_density_sync)
    
    arr /= np.max(arr) # Normalize
    return arr

def witek_syncopation(pattern, type):
    arr = np.array([0.0 for x in range(16)], dtype=float)
    pattern_LMH_count = get_lmh_counts(pattern)
    pattern_LMH_count = np.array(pattern_LMH_count, dtype=float)
    true_sync_salience = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1]
    sync_strength = [0,1,0,2, 0,1,0,3, 0,1,0,2, 0,1,0,4] #f 
    sync_strength = [0,4,0,1, 0,2,0,1, 0,3,0,1, 0,2,0,1] #b


    for i in range(16):
        # Code from sync part of flattening algorithm above
        if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
            note_values = [pattern_LMH_count[0][i], pattern_LMH_count[1][i], pattern_LMH_count[2][i]]
            for note in range(len(note_values)): # this loop removes multi hits
                if note_values[note]>=1:
                    note_values[note] = 1
            note_values_density_sync = np.array([note_values[0],note_values[1],note_values[2]], dtype=float)

            if i>0:
                if(i<len(pattern_LMH_count)):
                    if(true_sync_salience[i-1]>true_sync_salience[i]): # if note is syncop
                        if(pattern_LMH_count[0][i-1]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][i-1]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][i-1]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])
            """ if(i<15):
                if(true_sync_salience[i]>true_sync_salience[i+1]): # if note is syncop
                    if(pattern_LMH_count[0][i+1]==0):
                        note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                    if(pattern_LMH_count[1][i+1]==0):
                        note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                    if(pattern_LMH_count[2][i+1]==0):
                        note_values_density_sync[2] += (note_values[2]*sync_strength[i])
            if(i==15):
                    if(pattern_LMH_count[0][0]==0):
                        note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                    if(pattern_LMH_count[1][0]==0):
                        note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                    if(pattern_LMH_count[2][0]==0):
                        note_values_density_sync[2] += (note_values[2]*sync_strength[i]) """
            if type==2:
                note_values_density_sync[0] *= 3 # low
                note_values_density_sync[1] *= 2 # mid
            
            note_values_density_sync += 15.0
            note_values_density_sync /= 30.0 # witek math
            arr[i] = np.sum(note_values_density_sync)

    arr /= np.max(arr) # Normalize
    return arr

def metrical_strength(pattern, type):
    arr = np.array([0.0 for x in range(16)], dtype=float)
    pattern_LMH_count = get_lmh_counts(pattern)
    pattern_LMH_count = np.array(pattern_LMH_count, dtype=float)
    metric_sal_strength = [4,0,1,0, 2,0,1,0, 3,0,1,0, 2,0,1,0]

    for i in range(16):
        if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
            note_values = [pattern_LMH_count[0][i], pattern_LMH_count[1][i], pattern_LMH_count[2][i]]
            for note in range(len(note_values)): # this loop removes multi hits
                if note_values[note]>=1:
                    note_values[note] = 1
            note_values_density_sync_meter = np.array([note_values[0],note_values[1],note_values[2]], dtype=float)

            # [2] Normalized Density Salience, Metric Salience, Syncopation Strength
            if(i<15):
                if(metric_sal_strength[i]>metric_sal_strength[i+1]): # if meter is reinforced
                    if(pattern_LMH_count[0][i+1]==0):
                        note_values_density_sync_meter[0] += (note_values[0]*metric_sal_strength[i])
                    if(pattern_LMH_count[1][i+1]==0):
                        note_values_density_sync_meter[1] += (note_values[1]*metric_sal_strength[i])
                    if(pattern_LMH_count[2][i+1]==0):
                        note_values_density_sync_meter[2] += (note_values[2]*metric_sal_strength[i])
            if type==2:
                note_values_density_sync_meter[0] *= 3 # low
                note_values_density_sync_meter[1] *= 2 # mid
            arr[i] = np.sum(note_values_density_sync_meter)

    arr /= np.max(arr) # Normalize
    return arr

def relative_density(pattern, type):
    arr = np.array([0.0 for x in range(16)], dtype=float)
    pattern_LMH = get_LMH(pattern) # LOW MID HIGH
    pattern_LMH_count=[[0 for x in range(len(pattern_LMH))] for y in range(3)]
    total_count = [0.0 for x in range(4)]

    # Count multi-hits in same channel on step
    for i in range(len(pattern_LMH)):
        for j in range(len(pattern_LMH[i])):
            pattern_LMH_count[0][i] += 1 if pattern_LMH[i][j]==1 else 0 # LOW
            pattern_LMH_count[1][i] += 1 if pattern_LMH[i][j]==2 else 0 # MID
            pattern_LMH_count[2][i] += 1 if pattern_LMH[i][j]==3 else 0 # HIGH
    for i in range(3): # GET TOTAL COUNT
        total_count[i] = float(np.sum(pattern_LMH_count[i]))
        total_count[3] += float(np.sum(pattern_LMH_count[i]))

    # Initialize variables for flattening
    density = [0.0 for x in range(3)]
    salience = [0.0 for x in range(4)]
    norm_salience = [0.0 for x in range(3)]
    for i in range(3):
        density[i] = 0.0 if total_count[3]==0 else float(total_count[i]/total_count[3])
        salience[i] = 0.0 if density[i]<=0.0 else float(1/density[i])
        salience[3] += salience[i]
        #print(salience)
    for i in range(3):
        norm_salience[i] = salience[i]/salience[3]
    
    # Loop through pattern
    for i in range(len(pattern_LMH)):
        if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
            note_values = [norm_salience[0]*pattern_LMH_count[0][i],norm_salience[1]*pattern_LMH_count[1][i],norm_salience[2]*pattern_LMH_count[2][i]]
        else:
            note_values=[0.0,0.0,0.0]
        if type==2:
            note_values[0] *= 3
            note_values[1] *= 2
        arr[i] = np.sum(note_values)
    arr /= np.max(arr) # Norm
    return arr

def flatten_type(pattern, density_type, sync_type, meter, f_weight):
    ###
    # density types:
    # 0 := 1 if a note
    # 1 := norm. density
    # 2 := rel. density

    # sync types:
    # 0 := none
    # 1 := forwards
    # 2 := backwards

    # meter:
    # 0 := none
    # 1 := GTTM

    arr = np.array([0.0 for x in range(16)], dtype=float)
    note_values = np.array([1.0,1.0,1.0], dtype=float)
    notevals = np.array([[0.0 for x in range(16)] for y in range(3)], dtype=float)

    pattern_LMH = get_LMH(pattern) # LOW MID HIGH
    pattern_LMH_count=[[0 for x in range(len(pattern_LMH))] for y in range(3)]
    total_count = [0.0 for x in range(4)]
    for i in range(len(pattern_LMH)):
        for j in range(len(pattern_LMH[i])):
            pattern_LMH_count[0][i] += 1 if pattern_LMH[i][j]==1 else 0 # LOW
            pattern_LMH_count[1][i] += 1 if pattern_LMH[i][j]==2 else 0 # MID
            pattern_LMH_count[2][i] += 1 if pattern_LMH[i][j]==3 else 0 # HIGH
    for i in range(3): # GET TOTAL COUNT
        total_count[i] = float(np.sum(pattern_LMH_count[i]))
        total_count[3] += float(np.sum(pattern_LMH_count[i]))

    # Initialize variables for flattening
    density = [0.0 for x in range(3)]
    salience = [0.0 for x in range(4)]
    norm_salience = [0.0 for x in range(3)]
    means = [0.0,0.0]
    maxes = [0.0,0.0]
    for i in range(3):
        density[i] = 0.0 if total_count[3]==0 else float(total_count[i]/total_count[3])
        salience[i] = 0.0 if density[i]<=0.0 else float(1/density[i])
        salience[3] += salience[i]
    for i in range(3):
        norm_salience[i] = salience[i]/salience[3]


    # Testing other onset density method
    if density_type==1:
        norm_salience = [3.0,2.0,1.0]
    if density_type==0:
        norm_salience = [1.0,1.0,1.0]

    for i in range(16):
        if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
            note_values = [norm_salience[0]*pattern_LMH_count[0][i],norm_salience[1]*pattern_LMH_count[1][i],norm_salience[2]*pattern_LMH_count[2][i]]

        #### FREQ WEIGHT ####
        if f_weight==1:
            note_values[0] *= 3
            note_values[1] *= 2
        notevals[0][i] = note_values[0]
        notevals[1][i] = note_values[1]
        notevals[2][i] = note_values[2]
        #arr[i] = np.sum(note_values)
    
    #####
    # notevals has step & freq-wise note values [low,mid,high]
    # arr has stepwise note values [sum]

    
    if sync_type==1: # Forward Syncopation (note preceding metrical rest)
        true_sync_salience = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1]
        sync_strength = [0,1,0,2, 0,1,0,3, 0,1,0,2, 0,1,0,4]

        for i in range(16):
            # Code from sync part of flattening algorithm above
            if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
                note_values = [notevals[0][i],notevals[1][i],notevals[2][i]]
                note_values_density_sync = [notevals[0][i],notevals[1][i],notevals[2][i]]

                """ if i>0:
                    if(i<16):
                        if(true_sync_salience[i-1]>true_sync_salience[i]): # if note is syncop
                            if(pattern_LMH_count[0][i-1]==0):
                                note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                            if(pattern_LMH_count[1][i-1]==0):
                                note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                            if(pattern_LMH_count[2][i-1]==0):
                                note_values_density_sync[2] += (note_values[2]*sync_strength[i]) """
                if(i<15):
                    if(true_sync_salience[i]>true_sync_salience[i+1]): # if note is syncop
                        if(pattern_LMH_count[0][i+1]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][i+1]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][i+1]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])
                            
                if(i==15):
                        if(pattern_LMH_count[0][0]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][0]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][0]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])

                arr[i] += np.sum(note_values_density_sync)

    if sync_type==2: # Backwards Syncopation (note following metrical rest)
        true_sync_salience = [5,1,2,1, 3,1,2,1, 4,1,2,1, 3,1,2,1]
        sync_strength = [0,4,0,1, 0,2,0,1, 0,3,0,1, 0,2,0,1]

        for i in range(16):
            # Code from sync part of flattening algorithm above
            if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
                note_values = [notevals[0][i],notevals[1][i],notevals[2][i]]
                note_values_density_sync = [notevals[0][i],notevals[1][i],notevals[2][i]]
                if i>0:
                    if(i<16):
                        if(true_sync_salience[i-1]>true_sync_salience[i]): # if note is syncop
                            if(pattern_LMH_count[0][i-1]==0):
                                note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                            if(pattern_LMH_count[1][i-1]==0):
                                note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                            if(pattern_LMH_count[2][i-1]==0):
                                note_values_density_sync[2] += (note_values[2]*sync_strength[i])
                """if(i<15):
                    if(true_sync_salience[i]>true_sync_salience[i+1]): # if note is syncop
                        if(pattern_LMH_count[0][i+1]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][i+1]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][i+1]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])
                            
                if(i==15):
                        if(pattern_LMH_count[0][0]==0):
                            note_values_density_sync[0] += (note_values[0]*sync_strength[i])
                        if(pattern_LMH_count[1][0]==0):
                            note_values_density_sync[1] += (note_values[1]*sync_strength[i])
                        if(pattern_LMH_count[2][0]==0):
                            note_values_density_sync[2] += (note_values[2]*sync_strength[i])"""
                arr[i] += np.sum(note_values_density_sync)

    if meter==1:
        metric_sal_strength = [4,0,1,0, 2,0,1,0, 3,0,1,0, 2,0,1,0]

        for i in range(16):
            if(pattern_LMH_count[0][i]>0 or pattern_LMH_count[1][i]>0 or pattern_LMH_count[2][i]>0):
                note_values = [notevals[0][i],notevals[1][i],notevals[2][i]]
                meter = [notevals[0][i],notevals[1][i],notevals[2][i]]

                if(i<15):
                    if(metric_sal_strength[i]>metric_sal_strength[i+1]): # if meter is reinforced
                        if(pattern_LMH_count[0][i+1]==0):
                            meter[0] += (note_values[0]*metric_sal_strength[i])
                        if(pattern_LMH_count[1][i+1]==0):
                            meter[1] += (note_values[1]*metric_sal_strength[i])
                        if(pattern_LMH_count[2][i+1]==0):
                            meter[2] += (note_values[2]*metric_sal_strength[i])
                arr[i] += np.sum(meter)
    arr /= np.max(arr) # Norm
    
    # Line below converts everything to semicontinuous
    # arr = np.where(arr>=np.mean(arr), arr, 0.0)
    return arr