import functions as fn
import numpy as np
import matplotlib.pyplot as plot


cal = np.loadtxt("C:\\Users\\Elise\\PycharmProjects\\DCPEncryption\\venv\\cal.txt")
cal = np.delete(cal, 0, 1)
rateSigma, distSigma = fn.statsSigmaRange(cal, 150, 2000, 1, 50, 50)
rateIndex = np.argmin(rateSigma)
errorIndex = np.argmin(distSigma)
print("Min error rate at", rateIndex, "with value", rateSigma[rateIndex],
      "\nMin mean error distance at", errorIndex, "with value", distSigma[errorIndex])
fifty = np.zeros(shape=50)
for i in range(0, 50):
    fifty[i] = i + 1
plot.scatter(fifty, rateSigma)
plot.title("Error rate as a function of sigma")
plot.show()
plot.clf()
plot.scatter(fifty, distSigma)
plot.title("Mean distance of error as a function of sigma")
plot.show()
