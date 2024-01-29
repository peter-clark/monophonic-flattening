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

_descriptors = True
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
num_predictions = 10
predictions = np.array([[0.0 for x in range(256)] for y in range(num_predictions)], dtype=float)
predictions2 = np.array([[[0.0 for x in range(16)] for y in range(len(all_pattlists))] for z in range(num_predictions)], dtype=float)
num_alt_flats = 10
alt_flats = np.array([[0.0 for x in range(256)] for y in range(num_alt_flats)], dtype=float)
alt_flats2 = np.array([[[0.0 for x in range(16)] for y in range(len(all_pattlists))] for z in range(num_alt_flats)], dtype=float)

for pattern in range(len(all_pattlists)):
    #print(all_names[pattern])
    p = get_LMH(all_pattlists[pattern])
    note_cnt = np.array([0 for x in range(16)], dtype=float)
    note_cnt3 = np.array([[0 for x in range(16)] for y in range(3)], dtype=float)
    for st in range(len(p)): # This computes note counts
        if pattern in test_patterns:
            rng=2  #count notes indicator
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
    sc1 = np.where(flat[1]==1, flat[0],flat[1])
    sc2 = np.where(flat[3]==1, flat[2],flat[3])
    all_flat[pattern] = flat
    for i in range(len(flat_by_alg)):
        flat_by_alg[i][pattern] = flat[i]
    
    _pred2 = False
    if _pred2:
        predictions2[0][pattern] = flatten.onset_density(all_pattlists[pattern])
        predictions2[1][pattern] = flatten.witek_scaling(all_pattlists[pattern])
        predictions2[2][pattern] = flatten.syncopation(all_pattlists[pattern], type=1)
        predictions2[3][pattern] = flatten.syncopation(all_pattlists[pattern], type=2)
        predictions2[4][pattern] = flatten.witek_syncopation(all_pattlists[pattern], type=1)
        predictions2[5][pattern] = flatten.witek_syncopation(all_pattlists[pattern], type=2)
        predictions2[6][pattern] = flatten.metrical_strength(all_pattlists[pattern], type=1)
        predictions2[7][pattern] = flatten.metrical_strength(all_pattlists[pattern], type=2)
        predictions2[8][pattern] = flatten.relative_density(all_pattlists[pattern], type=1)
        predictions2[9][pattern] = flatten.relative_density(all_pattlists[pattern], type=2)
    
    _alt_flat2 = False
    if _alt_flat2:
        f_weight=0
        alt_flats2[0][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=1, meter=0, f_weight=f_weight)
        alt_flats2[1][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=2, meter=0, f_weight=f_weight)
        alt_flats2[2][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=0, meter=1, f_weight=f_weight)
        alt_flats2[3][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=1, meter=1, f_weight=f_weight)
        alt_flats2[4][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=2, meter=1, f_weight=f_weight)

        den_type = 0
        alt_flats2[5][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=1, meter=0, f_weight=f_weight)
        alt_flats2[6][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=2, meter=0, f_weight=f_weight)
        alt_flats2[7][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=0, meter=1, f_weight=f_weight)
        alt_flats2[8][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=1, meter=1, f_weight=f_weight)
        alt_flats2[9][pattern] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=2, meter=1, f_weight=f_weight)


    # Get other predictions and alt flats
    for patt in range(len(test_patterns)):
        if pattern == test_patterns[patt]:
            # Stepwise Onset Density
            predictions[0][patt*16:(patt+1)*16] = flatten.onset_density(all_pattlists[pattern])
            #predictions2[0][patt] = flatten.onset_density(all_pattlists[pattern])

            # Onset Density fBand Weighted
            predictions[1][patt*16:(patt+1)*16] = flatten.witek_scaling(all_pattlists[pattern])
            #predictions2[1][patt] = flatten.witek_scaling(all_pattlists[pattern])

            # Simple Syncopation
            predictions[2][patt*16:(patt+1)*16] = flatten.syncopation(all_pattlists[pattern], type=1)
            #predictions2[2][patt] = flatten.syncopation(all_pattlists[pattern], type=1)

            # Simple Syncopation fBand Weighted
            predictions[3][patt*16:(patt+1)*16] = flatten.syncopation(all_pattlists[pattern], type=2)
            #predictions2[3][patt] = flatten.syncopation(all_pattlists[pattern], type=2)

            # Witek Syncopation
            predictions[4][patt*16:(patt+1)*16] = flatten.witek_syncopation(all_pattlists[pattern], type=1)
            #predictions2[4][patt] = flatten.witek_syncopation(all_pattlists[pattern], type=1)

            # Witek Syncopation fBand Weighted
            predictions[5][patt*16:(patt+1)*16] = flatten.witek_syncopation(all_pattlists[pattern], type=2)
            #predictions2[5][patt] = flatten.witek_syncopation(all_pattlists[pattern], type=2)

            # Metrical Strength
            predictions[6][patt*16:(patt+1)*16] = flatten.metrical_strength(all_pattlists[pattern], type=1)
            #predictions2[6][patt] = flatten.metrical_strength(all_pattlists[pattern], type=1)

            # Metrical Strength fBand Weighted
            predictions[7][patt*16:(patt+1)*16] = flatten.metrical_strength(all_pattlists[pattern], type=2)
            #predictions2[7][patt] = flatten.metrical_strength(all_pattlists[pattern], type=2)

            # Relative Onset Density
            predictions[8][patt*16:(patt+1)*16] = flatten.relative_density(all_pattlists[pattern], type=1)
            #predictions2[8][patt] = flatten.relative_density(all_pattlists[pattern], type=1)

            # Relative Onset Density (fBand Weighted)
            predictions[9][patt*16:(patt+1)*16] = flatten.relative_density(all_pattlists[pattern], type=2)
            #predictions2[9][patt] = flatten.relative_density(all_pattlists[pattern], type=2)

            ### ALT FLATS SECTIONS ###
            f_weight = 0
            # Onset Forwards Syncopation
            #print(flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=1, meter=0, f_weight=f_weight)) 
            alt_flats[0][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=1, meter=0, f_weight=f_weight)
            # Onset Backwards Syncopation
            alt_flats[1][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=2, meter=0, f_weight=f_weight)
            # Onset Meter
            alt_flats[2][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=0, meter=1, f_weight=f_weight)
            # Onset Forwards Sync and Meter
            alt_flats[3][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=1, meter=1, f_weight=f_weight)
            # Onset Backwards Sync and Meter
            alt_flats[4][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=1, sync_type=2, meter=1, f_weight=f_weight)

            den_type = 0
            # Relative Onset Forwards Syncopation 
            alt_flats[5][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=1, meter=0, f_weight=f_weight)
            # Relative Onset Backwards Syncopation
            alt_flats[6][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=2, meter=0, f_weight=f_weight)
            # Relative Onset Meter
            alt_flats[7][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=0, meter=1, f_weight=f_weight)
            # Relative Onset Forwards Sync and Meter
            alt_flats[8][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=1, meter=1, f_weight=f_weight)
            # Relative Onset Backwards Sync and Meter
            alt_flats[9][patt*16:(patt+1)*16] = flatten.flatten_type(all_pattlists[pattern], density_type=den_type, sync_type=2, meter=1, f_weight=f_weight)

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

#p#rint(alt_flats[0])
file = open(os.getcwd()+"/data/alt_flats.pkl", 'wb')
pickle.dump(alt_flats, file, -1)
file.close()

file = open(os.getcwd()+"/data/overall_note_density.pkl", 'wb')
pickle.dump(all_nc, file, -1)
file.close()

file = open(os.getcwd()+"/data/channel_note_density.pkl", 'wb')
pickle.dump(all_nc3, file, -1)
file.close()

print("Patterns have been flattened\n")
for i in range(2):
    print(notes[i])

## Send flattened patterns + embedding coordinates to model to train
#       (4 x 4) -> embeddings x patterns 
#       - save models once trained
""" model_dir = dir + "/models/"
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
                    #f.write(str(predicted_coords[i])+"\n") if i!=len(predicted_coords)-1 else f.write(str(predicted_coords[i])) """

force_predictions_names = ['OnsDen', 'OnsDen_fW', 'Sync', 'Sync_fW', 'WitekSync', 'WitekSync_fW', 'MtrStr_fBand', 'MtrStr_fBand_fW', 'RelativeOnsDen', 'RelativeOnsDen_fBand'] # meter is done by freq. channel
alt_flats_names = ['OnsDen_forwardsSync', 'OnsDen_backwardsSync', 'OnsDen_meter', 'OnsDen_forwardsSync_meter', 'OnsDen_backwardsSync_meter', 'RelOnsDen_forwardsSync', 'RelOnsDen_backwardsSync', 'RelOnsDen_meter', 'RelOnsDen_forwardsSync_meter', 'RelOnsDen_backwardsSync_meter']
type=0 # 0 cont, 1 semi, 2 disc
stats_full = []
if _pred2:
    if type==1:
        mv = np.mean(predictions2, axis=-1, keepdims=True)
        predictions2 = np.where(predictions2>=mv,predictions2, 0.0)
    if type==2:
        mv = np.mean(predictions2, axis=-1, keepdims=True)
        predictions2 = np.where(predictions2>=mv,1, 0.0)
    model_dir = dir + "/models/"
    for embed in embeddings:
        for pred in range(len(predictions2)):
            predicted_coords = []
            # Build model
            model_dir += (force_predictions_names[pred])
            print(force_predictions_names[pred]+"--------------")
            name = force_predictions_names[pred]
            predicted_coords, stats = NN.NN_pipeline(predictions2[pred], embed, _savemodels, model_dir)
            #predicted_coords = NN.NN_pipeline(predictions2[pred], embed, _savemodels, model_dir, True)
            stats = stats+[name]
            stats_full.append(stats)
            model_dir=dir + "/models/"
            if _savepredictions:
                with open(dir+"/predictions/"+force_predictions_names[pred]+".csv",'w') as f:
                    writer = csv.writer(f)                
                    for i in range(len(predicted_coords)):
                        writer.writerow(predicted_coords[i])
                        #f.write(str(predicted_coords[i])+"\n") if i!=len(predicted_coords)-1 else f.write(str(predicted_coords[i]))
if _alt_flat2:
    if type==1:
        mv = np.mean(alt_flats2, axis=-1, keepdims=True)
        alt_flats2 = np.where(alt_flats2>=mv,alt_flats2, 0.0)
    if type==2:
        mv = np.mean(alt_flats2, axis=-1, keepdims=True)
        alt_flats2 = np.where(alt_flats2>=mv,1, 0.0)
    model_dir = dir + "/models/"
    for embed in embeddings:
        for pred in range(len(alt_flats2)):
            predicted_coords = []
            # Build model
            model_dir += (alt_flats_names[pred])
            print(alt_flats_names[pred]+"--------------")
            name = alt_flats_names[pred]
            predicted_coords, stats = NN.NN_pipeline(alt_flats2[pred], embed, _savemodels, model_dir)
            #predicted_coords = NN.NN_pipeline(predictions2[pred], embed, _savemodels, model_dir, True)
            stats = stats+[name]
            stats_full.append(stats)
            model_dir=dir + "/models/"
            if _savepredictions:
                with open(dir+"/predictions/"+alt_flats_names[pred]+".csv",'w') as f:
                    writer = csv.writer(f)                
                    for i in range(len(predicted_coords)):
                        writer.writerow(predicted_coords[i])
                        #f.write(str(predicted_coords[i])+"\n") if i!=len(predicted_coords)-1 else f.write(str(predicted_coords[i]))

print('\n'.join(' '.join(map(str, row)) for row in stats_full))
with open(dir+"/predictions/ALL_STATS.csv",'w') as f:
                    writer = csv.writer(f)                
                    for i in range(len(stats_full)):
                        writer.writerow(stats_full[i])
print(f"Runtime: {time.perf_counter()-s:.2f} seconds")
