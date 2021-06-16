import numpy as np
import math
import matplotlib.pyplot as plt

data = np.load('btc_price.npy')
n = np.size(data)
I = np.identity(n)
D = np.empty((0, n))

for i in range(n-1):
    this_row = [[0]*n]
    this_row[0][i] = 1
    this_row[0][i+1] = -1
    D = np.append(D, np.array(this_row), axis=0)

L = 100
A = np.concatenate((I, math.sqrt(L) * D))
b = np.concatenate((data.reshape(n, 1), np.zeros((n-1, 1))))
AT = A.transpose()

left = np.dot(AT, A)
leftInverse = np.linalg.inv(left)
right = np.dot(AT, b)
x_hat = np.dot(leftInverse, right)

plt.plot(x_hat)
plt.show()
