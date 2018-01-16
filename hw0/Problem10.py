#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
mean = [0, 0]
cov = [[1, 0], [0, 1]]
result = np.random.multivariate_normal(mean, cov, 10)
x, y=result.T
print(result)
print(x)
print(y)
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()
