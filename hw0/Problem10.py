import matplotlib.pyplot as plt
import numpy as np
mean = [0, 0]
cov = [[1, 0], [0, 1]]
x, y = np.random.multivariate_normal(mean, cov, 1000)
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()