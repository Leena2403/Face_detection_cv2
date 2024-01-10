# Basics of numpy

import numpy as np
import matplotlib.pyplot as plt 

# random number fom 0-1
a = np.random.random()
print(a)

np.random.randint(0,10,size=(4,5))

data = np.random.randn(1000)
print(data.mean(), data.std())

b = np.random.randint(0,10,size=(5,6))
print(b)

print(b.sum())


print(b.sum(axis=0, keepdims=True).shape)
print(np.linspace(0,10,11))
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)

plt.plot(x,y)