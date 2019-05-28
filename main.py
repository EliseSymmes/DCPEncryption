import functions as fn
import numpy as np


x = np.array([[2, 3], [1, 1], [1, 2], [9, 9], [0, 1], [1, 1]])
s, a, sigmaStar = fn.keygen(50)
query = fn.genSetPoints(2, 1000, 0.0, 100.0)
neighborhood = fn.genSetPoints(2, 1000, 0.0, 100.0)
errorDists = fn.errorDistance(query, neighborhood, s, a, sigmaStar)
noZeros = fn.noZerosPlease(errorDists)
errorRate = len(noZeros) / len(errorDists)
meanError = np.mean(errorDists)
print("Error rate:", errorRate, "\nMean Error:", meanError)
