import pandas as pd
import numpy as np

a = np.array([1,3,5,2,7,4,2])

print(a)
print(a.shape)
df = pd.DataFrame(a)
df['1'] = range(7)
df_s = df.sort_values(0, ascending=False)
df = df_s.drop(0,axis=1)
b = df.values
print(b)
print(b.shape)