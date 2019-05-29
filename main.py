import functions as fn
import numpy as np


cal = np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\cal.txt")
cal = np.delete(cal, 0, 1)
meanError = np.zeros(shape=50)
errorRate = np.zeros(shape=50)
for i in range(0, 50):
    query, neighbors = fn.splitArr(cal, 150, 2000)
    s, a, sigmaStar = fn.keygen(50)
    errorDists = fn.errorDistance(query, neighbors, s, a, sigmaStar)
    noZeros = fn.noZerosPlease(errorDists)
    errorRate[i] = len(noZeros) / len(errorDists)
    meanError[i] = np.mean(errorDists)
    print("Trial", i, ": Error rate:", errorRate[i], "\n          Mean Error:", meanError[i], "\n s =", s, "  a =", a,
          "sigma* =", sigmaStar)
print("Mean Error rate:", np.mean(errorRate), "\nMean Mean Error:", np.mean(meanError))
