#!/usr/bin/python
import numpy as np
w, v = np.linalg.eig(np.array([[1, 0], [1, 3]]))
print(w)
print(v)
