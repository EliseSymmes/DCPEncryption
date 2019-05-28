import random
import functions as fn
import numpy as np

# print(fn.plotErrorDistance(1.6, 1.0, 100, 3, 10000))
# print(fn.plotErrorDistance(1.6, 2.0, 100, 3, 1000, 10000))

# fn.plotPDist(5000)
x = np.array([[2, 3], [1, 1], [1, 2], [9, 9], [0, 1], [1, 1]])
# alpha = 4.0
# print(fn.numberOverlap(x, alpha))
# fn.scatter2d(fn.genSetPoints(2, 20, 10), alpha=alpha, enc=True)
print(fn.internalNN(x))
