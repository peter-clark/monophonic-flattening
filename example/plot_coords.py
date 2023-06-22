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

x = np.array(c.X)
y = np.array(c.Y)
x_max = np.max(x)
x_min = np.min(x)
y_max = np.max(y)
y_min = np.min(y)
print(f"max({x_max},{y_max}), min({x_min},{y_min})")
sections=[[]for x in range(16)]
indices=[[]for x in range(17)]
'''
[12][13][14][15]
[08][09][10][11]
[04][05][06][07]
[00][01][02][03]
'''
for i in range(len(x)):
    if y[i]<=0.25: # row 1
        if x[i]<=0.25: # col 1
            sections[0].append([x[i],y[i]])
            indices[0].append(i)
        elif x[i]<=0.50 and x[i]>0.25: # col 2
            sections[1].append([x[i],y[i]])
            indices[1].append(i)
        elif x[i]<=0.75 and x[i]>0.5:
            sections[2].append([x[i],y[i]])
            indices[2].append(i)
        else:
            sections[3].append([x[i],y[i]])
            indices[3].append(i)
    elif y[i]<=0.5: # row 2
        if x[i]<=0.25: # col 1
            sections[4].append([x[i],y[i]])
            indices[4].append(i)
        elif x[i]<=0.50 and x[i]>0.25:
            sections[5].append([x[i],y[i]])
            indices[5].append(i)
        elif x[i]<=0.75 and x[i]>0.5:
            sections[6].append([x[i],y[i]])
            indices[6].append(i)
        else:
            sections[7].append([x[i],y[i]])
            indices[7].append(i)
    elif y[i]<=0.75:
        if x[i]<=0.25: # col 1
            sections[8].append([x[i],y[i]])
            indices[8].append(i)
        elif x[i]<=0.50 and x[i]>0.25:
            sections[9].append([x[i],y[i]])
            indices[9].append(i)
        elif x[i]<=0.75 and x[i]>0.5:
            sections[10].append([x[i],y[i]])
            indices[10].append(i)
        else:
            sections[11].append([x[i],y[i]])
            indices[11].append(i)
    else:
        if x[i]<=0.25: # col 1
            sections[12].append([x[i],y[i]])
            indices[12].append(i)
        elif x[i]<=0.50 and x[i]>0.25:
            sections[13].append([x[i],y[i]])
            indices[13].append(i)
        elif x[i]<=0.75 and x[i]>0.5:
            sections[14].append([x[i],y[i]])
            indices[14].append(i)
        else:
            sections[15].append([x[i],y[i]])
            indices[15].append(i)
n=0
t=0
for i in range(16):
    print(f"[{t},{i%4}]: {len(sections[i])}")
    n += len(sections[i])
    if (i+1)%4==0 and i!=0:
        t+=1
print(n)
""" for i in range(1513):
    # set c1 coords to mds (no line) if distance too high
    x[1][i] = x[0][i] if EuclideanDistance([x[0][i],y[0][i]],[x[1][i],y[1][i]])<=0.15 else x[1][i]
    y[1][i] = y[0][i] if EuclideanDistance([x[0][i],y[0][i]],[x[1][i],y[1][i]])<=0.15 else y[1][i]
 """
plt.scatter(x,y, c='orange',s=7)
plt.scatter(0.46644574, 0.40844336, c='green',s=12)
#box
plt.plot((1,1), (0,1), c='grey') #right
plt.plot((1,0), (0,0), c='grey') #bottom
plt.plot((1,0), (1,1), c='grey') #top
plt.plot((0,0), (1,0), c='grey') #left
#verts
plt.plot((0.25,0.25), (0,1), c='grey') #25%
plt.plot((0.5,0.5), (0,1), c='grey') #50%
plt.plot((0.75,0.75), (0,1), c='grey') #75%
#horiz
plt.plot((0,1),(0.25,0.25), c='grey') #25%
plt.plot((0,1),(0.5,0.5), c='grey') #50%
plt.plot((0,1),(0.75,0.75), c='grey') #75%

#plt.plot((x_min,y_min),(x_min,0), c='grey')
#plt.plot((x_min,y_min),(x_max,0), c='grey')

plt.show()