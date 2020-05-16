import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1= pd.read_csv('exp/trust.csv',header=None)
df2= pd.read_csv('exp/trust1.csv',header=None)
df3= pd.read_csv('exp/trust2.csv',header=None)

df1 = df1.dropna()
df2 = df2.dropna()
df3 = df3.dropna()

tag1 = df1.values
tag2 = df2.values
tag3 = df3.values
print('ndcg_mean')
print(np.mean(tag1))
print(np.mean(tag2))
print(np.mean(tag3))

a = np.zeros(len(tag1))
b = np.zeros(len(tag1))
t = len(a)
for i in range(t):
    a[i] = abs( tag1[i] - tag2[i] )
    b[i] = abs( tag1[i] - tag3[i] )
print('MAN')
print(np.mean(a))
print(np.mean(b))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)


ax.hist((tag1[:,0],tag2[:,0],tag3[:,0]),color=['red','blue','green'],label=['normal','E1','E2'])
ax.legend(loc='upper left')
ax.set_xlabel('NDCG')
ax.set_ylabel('number of users')
plt.show()

'''

#boxplot
ndcg = (tag1,tag2,tag3)

fig, ax = plt.subplots()

bp = ax.boxplot(ndcg,whis='range')
ax.set_xticklabels(['a','b','c'])


plt.xlabel('model')
plt.ylabel('ndcg')

plt.ylim([0,1])
plt.grid()

plt.show()
'''