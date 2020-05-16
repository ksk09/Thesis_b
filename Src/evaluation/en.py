import matplotlib.pyplot as plt
import numpy as np

x1 = np.array([38.6,37.2,24.2])
x2 = np.array([50.7,49.3,0.0])
x3 = np.array([37.5,36.2,26.2])
label = ['u','m','t']
plt.pie(x3,labels=label,autopct='%1.1f%%')
plt.show()