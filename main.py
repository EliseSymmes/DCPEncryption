import functions as fn
import numpy as np
import matplotlib.pyplot as plot


cal = np.delete(np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\cal.txt"), 0, 1)
iris = np.delete(np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\iris.txt"), 0, 1)
querySize = 25
neighborSize = 125
sigmaLower = 20
sigmaUpper = 2000
trialsPerSigma = 200
step = 20
rateSigma, distSigma = fn.statsSigmaRange(iris, querySize, neighborSize, sigmaLower, sigmaUpper, trialsPerSigma, step)
rateIndex = np.argmin(rateSigma)
errorIndex = np.argmin(distSigma)
axis = np.zeros(shape=len(rateSigma))
for i in range(0, len(rateSigma)):
    axis[i] = sigmaLower + i * step
print("Min error rate at", axis[rateIndex], "with value", rateSigma[rateIndex],
      "\nMin mean error distance at", axis[errorIndex], "with value", distSigma[errorIndex])
plot.scatter(axis, rateSigma)
plot.title("Error rate as a function of sigma")
plot.show()
plot.clf()
plot.scatter(axis, distSigma)
plot.title("Mean distance of error as a function of sigma")
plot.show()
