import functions as fn
import numpy as np
import matplotlib.pyplot as plot


cal = np.delete(np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\cal.txt"), 0, 1)
querySize = 150
neighborSize = 2000
sigmaLower = 1
sigmaUpper = 50
trials = 50
rateSigma, distSigma = fn.statsSigmaRange(cal, querySize, neighborSize, sigmaLower, sigmaUpper, trials)
rateIndex = np.argmin(rateSigma)
errorIndex = np.argmin(distSigma)
print("Min error rate at", rateIndex, "with value", rateSigma[rateIndex],
      "\nMin mean error distance at", errorIndex, "with value", distSigma[errorIndex])
axis = np.zeros(shape=50)
for i in range(0, 50):
    axis[i] = i + 1
plot.scatter(axis, rateSigma)
plot.title("Error rate as a function of sigma")
plot.show()
plot.clf()
plot.scatter(axis, distSigma)
plot.title("Mean distance of error as a function of sigma")
plot.show()
