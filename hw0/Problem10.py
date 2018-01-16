#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
mean = [0, 0]
cov = [[1, 0], [0, 1]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y, 'x')
plt.axis([-6,6,-6,6])
plt.savefig("Problem10-a.pdf")
plt.clf()

mean = [1,1]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y, 'x')
plt.axis([-6,6,-6,6])
plt.savefig("Problem10-b.pdf")
plt.clf()

mean = [0,0]
cov = [[2, 0], [0, 2]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y, 'x')
plt.axis([-6,6,-6,6])
plt.savefig("Problem10-c.pdf")
plt.clf()

cov = [[1, 0.5], [0.5, 1]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y, 'x')
plt.axis([-6,6,-6,6])
plt.savefig("Problem10-d.pdf")
plt.clf()

cov = [[1, -0.5], [-0.5, 1]]
result = np.random.multivariate_normal(mean, cov, 1000)
x, y=result.T
plt.plot(x, y, 'x')
plt.axis([-6,6,-6,6])
plt.savefig("Problem10-e.pdf")
plt.clf()
