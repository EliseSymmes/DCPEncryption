import functions as fn
import numpy as np
import matplotlib.pyplot as plot


cal = np.delete(np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\cal.txt"), 0, 1)
iris = np.delete(np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\iris.txt"), 0, 1)
querySize = 25
neighborSize = 125
neighborSizes = fn.makeArr(25, 10)
sigmaLower = 0.1
sigmaUpper = 3
trials = 50
step = 0.1
ratesNeigh, distsNeigh, betterNeigh = fn.statsNeighborRange(cal, 50, 25, neighborSizes, trials)
ratesMean = np.zeros(shape=len(neighborSizes))
distsMean = np.zeros(shape=len(neighborSizes))
betterMean = np.zeros(shape=len(neighborSizes))
for i in range(0, len(neighborSizes)):
    ratesMean[i] = np.mean(ratesNeigh[i])
    distsMean[i] = np.mean(distsNeigh[i])
    betterMean[i] = np.mean(fn.noZerosPlease(betterNeigh[i]))
plot.scatter(neighborSizes, ratesMean)
plot.title("Error rate as a function of neighbor size")
plot.show()
plot.clf()
plot.scatter(neighborSizes, betterMean)
plot.title("Mean preferred neighbors as a function of neighbor size")
plot.axhline(y=1)
plot.show()
# rateSigma, distSigma = fn.statsSigmaRange(iris, querySize, neighborSize, sigmaLower, sigmaUpper, trials, step)
# rateIndex = np.argmin(rateSigma)
# errorIndex = np.argmin(distSigma)
# axis = np.zeros(shape=len(rateSigma))
# for i in range(0, len(rateSigma)):
#     axis[i] = sigmaLower + i * step
# plot.scatter(axis, rateSigma)
# plot.title("Error rate as a function of sigma")
# plot.show()
# plot.clf()
# plot.scatter(axis, distSigma)
# plot.title("Mean distance of error as a function of sigma")
# plot.show()
