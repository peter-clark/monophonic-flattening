import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

def EuclideanDistance(a, b):
    d = np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
    return d

coords_dir = os.getcwd()+"/embeddings/mds.csv"
pred_dir = os.getcwd()+"/predictions/continuous2.csv"
c = pd.read_csv(coords_dir)
p = pd.read_csv(pred_dir)
c.head()
p.head()
""" c.plot(kind='scatter',x='MDS_X',y='MDS_Y')
p.plot(kind='scatter',x='CONT1_X',y='CONT1_Y')
plt.show() """


all_dir = os.getcwd()+"/all.csv"
all = pd.read_csv(all_dir)
x = np.array([all.MDS_X, p.X])
y = np.array([all.MDS_Y,p.Y])
for i in range(1513):
    # set c1 coords to mds (no line) if distance too high
    x[1][i] = x[0][i] if EuclideanDistance([x[0][i],y[0][i]],[x[1][i],y[1][i]])<=0.15 else x[1][i]
    y[1][i] = y[0][i] if EuclideanDistance([x[0][i],y[0][i]],[x[1][i],y[1][i]])<=0.15 else y[1][i]

print(all.head())
#print(all.info())
plt.scatter(all.MDS_X, all.MDS_Y, c='blue',s=7)
plt.scatter(p.X,p.Y, c='orange',s=7)
plt.scatter(0.46644574, 0.40844336, c='green',s=12)
plt.plot(x, y, c='grey')

plt.show()