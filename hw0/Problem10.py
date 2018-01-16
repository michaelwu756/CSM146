#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
mean = [0, 0]
cov = [[1, 0], [0, 1]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y)
plt.axis('equal')
plt.savefig("Problem10-a.png")
plt.clf

mean = [1,1]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y)
plt.axis('equal')
plt.savefig("Problem10-b.png")
plt.clf

mean = [0,0]

result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y)
plt.axis('equal')
plt.savefig("Problem10-b.png")

mean = [1,1]
cov = [[2, 0], [0, 2]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y)
plt.axis('equal')
plt.savefig("Problem10-c.png")
plt.clf

cov = [[1, 0.5], [0.5, 1]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y)
plt.axis('equal')
plt.savefig("Problem10-d.png")
plt.clf

cov = [[1, -0.5], [-0.5, 1]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y)
plt.axis('equal')
plt.savefig("Problem10-d.png")
plt.clf