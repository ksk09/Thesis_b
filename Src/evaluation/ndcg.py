import pandas as pd
import numpy as np
'''
df1= pd.read_csv('exp/movielen20.csv',header=None)
df2= pd.read_csv('exp/movielen20_e1_inf.csv',header=None)
df3= pd.read_csv('exp/movielen20_e2_inf.csv',header=None)

'''
df1= pd.read_csv('exp/trust.csv',header=None)
df2= pd.read_csv('exp/trust1.csv',header=None)
df3= pd.read_csv('exp/trust2.csv',header=None)

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()

df = pd.concat([df1,df2,df3],axis=1)
max_tag = df.values

x = 0
y = 0
z = 0

nd1 = 0
nd2 = 0
nd2_1 = 0
nd3 = 0
nd3_1 = 0
ndcg = 0
t = max_tag.shape[0]

for i in range(t):
    ndcg += max_tag[i,0]
    if np.argmax(max_tag[i]) == 0:
        x += 1
        nd1 += max_tag[i,0]
    elif np.argmax(max_tag[i]) == 1:
        y += 1
        nd2 += max_tag[i,1]
        nd2_1 += max_tag[i,0]
    elif np.argmax(max_tag[i]) == 2:
        z += 1
        nd3 += max_tag[i,2]
        nd3_1 += max_tag[i,0]

print(x,y,z)

print(ndcg/t, nd1/x, nd2/y, nd3/z)
print(nd2_1/y, nd3_1/z)