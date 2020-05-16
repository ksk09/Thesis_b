import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df1= pd.read_csv('exp/div.csv',header=None)
df2= pd.read_csv('exp/div_e1.csv',header=None)
df3= pd.read_csv('exp/div_e2.csv',header=None)

tag1 = df1.values
tag2 = df2.values
tag3 = df3.values
a = np.zeros(610)
b = np.zeros(610)
for i in range(610):
    a[i] = 20 -len(set(tag1[i]) & set(tag2[i]))
    b[i] = 20 - len(set(tag1[i]) & set(tag3[i]))

print(np.mean(b))
