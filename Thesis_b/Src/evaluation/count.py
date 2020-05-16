import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1= pd.read_csv('exp/trust.csv',header=None)
df2= pd.read_csv('exp/trust1.csv',header=None)
df3= pd.read_csv('exp/trust2.csv',header=None)



df = pd.concat([df1,df2,df3],axis=1)
max_tag = df.values

x = 0
y = 0
z = 0

t = max_tag.shape[0]
for i in range(t):
    if np.argmax(max_tag[i]) == 0:
        x += 1
    elif np.argmax(max_tag[i]) == 1:
        y += 1
    elif np.argmax(max_tag[i]) == 2:
        z += 1


print(x,y,z)
