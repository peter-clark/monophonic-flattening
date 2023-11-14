import os
import sys
import numpy as np
import functions as fun
import descriptors as desc
import flatten as flatten
import neuralnetwork as NN
import time
import csv
import pickle
s = time.perf_counter()

## Initalize variables
dir = os.getcwd()
_savepatterns = False
_saveflattened = False
_saveembeddings = False
_savemodels = False
_savepredictions = False
#_savepredictions = _savemodels

## Extract all patterns and names from MIDI files in (+sub)folder
all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments")
patterns_dir = dir + "/patterns/"
print("\nExtracted Patterns\n")

if _savepatterns:
    for index in range(len(all_pattlists)):
        filename = patterns_dir+str(all_names[index])+".txt"
        with open(filename, "w") as f:
            f.write(str(all_pattlists[index]))

_descriptors = False
if _descriptors:
    ## Get polyphonic descriptors for patterns
    d = desc.lopl2descriptors(all_pattlists)

    ## Slice for 5 significant descriptors
    _d = np.asarray([np.asarray([de[2],de[3],de[7],de[8],de[13]]) for de in d])

    print("Calculated Polyphonic Descriptors \n")


    ## Get coordinates from embedding done on poly-descriptors
    #       [MDS, PCA, TSNE, UMAP]
    embeddings_dir = dir + "/embeddings/"
    embeddings = []
    embeddings_names = ["MDS","PCA","TSNE","UMAP"]

    mds_pos = fun.d_mtx_2_mds(d)
    embeddings.append(mds_pos)
    print("Got embedding coordinates\n")

#   Save if desired
if _saveembeddings:
    for i in range(len(embeddings)):
        filename = embeddings_dir + embeddings_names[i] + ".txt"
        filename_csv = embeddings_dir + embeddings_names[i] + ".csv"
        with open(filename,"w") as f:
            g=open(filename_csv,'w')
            writer = csv.writer(g)
            for pos in range(len(embeddings[i])):
                writer.writerow(embeddings[i][pos])
                f.write(str(embeddings[i][pos])+"\n") if pos != len(embeddings[i]-1) else f.write(str(embeddings[i][pos]))
            g.close()
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

            

## Apply flattening algorithms to all patterns
#       [continuous1, continuous2, discrete1, discrete2]
flat_names = ["continuous1", "discrete1", "continuous2", "discrete2","semicontinuous1", "semicontinuous2"]
flattened_dir = dir + "/flattened/"
all_flat = [[] for x in range(len(all_pattlists))]
flat_by_alg = [[[] for x in range(len(all_pattlists))] for y in range(6)]
notes = np.array([[0, 0, 0, 0] for y in range(2)], dtype=float) # notes / rests
test_patterns = [894, 423, 1367, 249, 939, 427, 590, 143, 912, 580, 1043, 673, 1359, 736, 678, 1355]
all_nc = np.array([], dtype=float)
all_nc3 = np.array([], dtype=float)

# For comparing literature predictions and results
num_predictions = 8
predictions = np.array([[0.0 for x in range(256)] for y in range(num_predictions)], dtype=float)

for pattern in range(len(all_pattlists)):
    #print(all_names[pattern])
    p = get_LMH(all_pattlists[pattern])
    note_cnt = np.array([0 for x in range(16)], dtype=float)
    note_cnt3 = np.array([[0 for x in range(16)] for y in range(3)], dtype=float)
    for st in range(len(p)): # This computes note counts
        if pattern in test_patterns:
            rng=2
        else:
            rng=1
        for i in range(rng):
            if len(p[st])==0:
                notes[i][0]+=1
                note_cnt[st]=0
                note_cnt3[:,st]=0
            else:
                for n in range(len(p[st])):
                    note_cnt[st]+=1
                    if p[st][n]==1:
                        notes[i][1]+=1
                        note_cnt3[0][st]+=1
                    elif p[st][n]==2:
                        notes[i][2]+=1
                        note_cnt3[1][st]+=1
                    else:
                        notes[i][3]+=1
                        note_cnt3[2][st]+=1
    note_cnt /= np.max(note_cnt)
    sumsal=0
    ac3=np.array([0.0 for x in range(16)], dtype=float)
    for i in range(3):
        if np.sum(note_cnt3[i]>0):
            sumsal += (1 / ((np.sum(note_cnt3[i])) / (np.sum(note_cnt))))
    for i in range(3):
        # Relative Onset Density
        #if np.sum(note_cnt3[i]>0):
        #    note_cnt3[i] *= ((1 / ( (np.sum(note_cnt3[i])) / (np.sum(note_cnt)) ) ) / float(sumsal))
        
        # Witek scaling
        if i==0:
            note_cnt3[i] *= 3 # low freq bias
        elif i==1:
            note_cnt3[i] *= 2 # mid freq bias
        if rng==2:
            ac3 += note_cnt3[i]
    if rng==2:
        all_nc = np.append(all_nc, note_cnt, axis=0)
        ac3 /= np.max(ac3)
        all_nc3 = np.append(all_nc3, ac3)
    #print(f"{pattern}:{note_cnt}\n{note_cnt3}")             
    flat = flatten.flat_from_patt(all_pattlists[pattern])
    #print(len(flat))
    #print(len(flat[1]))
    sc1 = np.where(flat[1]==1, flat[0],flat[1])
    sc2 = np.where(flat[3]==1, flat[2],flat[3])
    #flat.append(sc1)
    #flat.append(sc2)
    all_flat[pattern] = flat
    for i in range(len(flat_by_alg)):
        flat_by_alg[i][pattern] = flat[i]
    
    # Get other predictions
    for patt in range(len(test_patterns)):
        if pattern == test_patterns[patt]:
            # Stepwise Onset Density
            predictions[0][patt*16:(patt+1)*16] = flatten.onset_density(all_pattlists[pattern])

            # Onset Density fBand Weighted
            predictions[1][patt*16:(patt+1)*16] = flatten.witek_scaling(all_pattlists[pattern])

            # Simple Syncopation
            predictions[2][patt*16:(patt+1)*16] = flatten.syncopation(all_pattlists[pattern], type=1)

            # Simple Syncopation fBand Weighted
            predictions[3][patt*16:(patt+1)*16] = flatten.syncopation(all_pattlists[pattern], type=2)

            # Witek Syncopation
            predictions[4][patt*16:(patt+1)*16] = flatten.witek_syncopation(all_pattlists[pattern], type=1)

            # Witek Syncopation fBand Weighted
            predictions[5][patt*16:(patt+1)*16] = flatten.witek_syncopation(all_pattlists[pattern], type=2)

            # Metrical Strength
            predictions[6][patt*16:(patt+1)*16] = flatten.metrical_strength(all_pattlists[pattern], type=1)

            # Metrical Strength fBand Weighted
            predictions[7][patt*16:(patt+1)*16] = flatten.metrical_strength(all_pattlists[pattern], type=2)
            print(f"({patt}/16)")

#   Save if desired
    if _saveflattened:
        filename = flattened_dir+str(all_names[pattern])+".txt"
        with open(filename, "w") as f:
            for i in range(len(flat)):
                f.write(str(all_flat[pattern][i])+"\n") if i!=len(flat)-1 else f.write(str(all_flat[pattern][i]))
        for i in range(len(flat_by_alg)):
            with open(dir+"/flat/"+flat_names[i]+".txt",'w') as g:
                for pattern in range(len(all_pattlists)):
                    g.write(str(flat_by_alg[i][pattern])+"\n") if i!=len(all_pattlists)-1 else g.write(str(flat_by_alg[i][pattern]))
file = open(os.getcwd()+"/flat/flatbyalg.pkl", 'wb')
pickle.dump(flat_by_alg, file, -1)
file.close()

file = open(os.getcwd()+"/data/force_predictions.pkl", 'wb')
pickle.dump(predictions, file, -1)
file.close()

file = open(os.getcwd()+"/data/overall_note_density.pkl", 'wb')
pickle.dump(all_nc, file, -1)
file.close()

file = open(os.getcwd()+"/data/channel_note_density.pkl", 'wb')
pickle.dump(all_nc3, file, -1)
file.close()

print("Patterns have been flattened\n")
for i in range(2):
    print(notes/np.sum(notes[i]))

## Send flattened patterns + embedding coordinates to model to train
#       (4 x 4) -> embeddings x patterns 
#       - save models once trained
model_dir = dir + "/models/"
for embed in embeddings:
    for alg in range(len(flat_by_alg)):
        predicted_coords = []
        # Build model
        model_dir += (flat_names[alg])
        print(flat_names[alg]+"--------------")
        predicted_coords = NN.NN_pipeline(flat_by_alg[alg], embed, _savemodels, model_dir)
        #predicted_coords = NN.NN_pipeline(flat_by_alg[alg], embed, _savemodels, model_dir, True)
        model_dir=dir + "/models/"
        if _savepredictions:
            with open(dir+"/predictions/"+flat_names[alg]+".csv",'w') as f:
                writer = csv.writer(f)                
                for i in range(len(predicted_coords)):
                    writer.writerow(predicted_coords[i])
                    #f.write(str(predicted_coords[i])+"\n") if i!=len(predicted_coords)-1 else f.write(str(predicted_coords[i]))

print(f"Runtime: {time.perf_counter()-s:.2f} seconds")
