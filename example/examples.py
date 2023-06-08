#examples

import os
import functions as fun
import descriptors as desc
import flatten as flatten
import to_pd as puredata
import neuralnetwork as NN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import csv
start = time.perf_counter()

# prepare the rhythm space 
##########################

# parse all midi patterns found in a folder (including subfolders)
all_pattlists, all_names = fun.rootfolder2pattlists("midi/","allinstruments")

# convert themn to the pattlist format and get their names
d = desc.lopl2descriptors(all_pattlists) 

# create positions from descriptors using embedding
pos = fun.d_mtx_2_mds(d)

# create delaunay triangulations
triangles = fun.create_delaunay_triangles(pos) #create triangles
print(f" Triangles: {len(triangles)} {len(triangles[0])}")
print(triangles)
# define the number of cols and rows of the space
hash_density = 2

# hash the space for faster searches. groups triangles in hash categories
hashed_triangles = fun.hash_triangles(pos, triangles, hash_density) # 2 means divide the space in 2 cols and 2 rows
print(f"HASH TRIANGLES {len(hashed_triangles)} {len(hashed_triangles[0])}")
print(hashed_triangles)
# search the rhythm space
#########################
p = np.random.randint(0,1513) # get random pattern
print(p)


input_patt = all_pattlists[p]
patt_name = all_names[p]

input_flat = flatten.flat_from_patt(input_patt)
flat_by_alg = [[] for y in range(4)]
for i in range(len(flat_by_alg)):
        flat_by_alg[i] = input_flat[i]
#print(flat_by_alg)
""" for i in input_flat:
    print(i) """
pred_dir = os.getcwd()+"/predictions/continuous1.csv"
preds = pd.read_csv(pred_dir)
pred_coords = np.array([preds.X[p],preds.Y[p]])
coords_dir = os.getcwd()+"/embeddings/mds.csv"
c = pd.read_csv(coords_dir)
coords = np.array([c.X[p],c.Y[p]])
model_dir = os.getcwd()+"/models/continuous2.pt"

model = NN.build_model()
model.load_state_dict(NN.torch.load(model_dir))
pred_coords = model(NN.torch.Tensor(input_flat[2]).float()).detach().numpy()


#pred_coords_all = NN.NN_pipeline(flat_by_alg[0], pos, False, model_dir, True)
#pred_coords = pred_coords_all[p]


s = [0.5,0.5] # coordinates to search

print(f"searched for pattern/predicted coordinates: , {coords}-{pred_coords} --> {puredata.EuclideanDistance(coords,pred_coords)}")

output_patt = fun.position2pattern(pred_coords, all_pattlists,  pos, triangles, hashed_triangles, hash_density)
input_patt = puredata.parse_8(input_patt)
output_patt = puredata.parse_8(output_patt)
print(f"searched for pattern/predicted coordinates: , {coords}-{pred_coords} --> {puredata.EuclideanDistance(coords,pred_coords)}")
#print("original pattern:", input_patt)
print("obtained pattern:", output_patt)

puredata.send_poly_patt(input_patt,patt_name)
puredata.send_flat_patts(input_flat)
puredata.send_poly_patt_predicted(output_patt, patt_name)


############# plot
#plt.scatter(mds_pos[:,0], mds_pos[:,1], color="0", s=5, marker="o", alpha=0.5, edgecolors="none")
#plt.xlim(-2,2)
#plt.ylim(-2,2)
#plt.title("rhythmspace")
#plt.gca().set_aspect(1)
#plt.show()

## Show Runtime
print(f"Runtime: {time.perf_counter()-start:.2f} seconds")
