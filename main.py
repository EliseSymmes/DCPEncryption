import functions as fn
import numpy as np
import matplotlib.pyplot as plt


# iris = np.delete(np.loadtxt("iris.txt"), 0, 1)
x = [[1, 1], [1, 2], [0, 1], [2, 2], [9, 9]]
cal = np.delete(np.loadtxt("cal.txt"), 0, 1)
river = np.loadtxt('river.csv', delimiter=',', usecols=(0, 1), skiprows=1, dtype=type(1.1))
mili = np.loadtxt('military.csv', delimiter=',', usecols=(0, 1), skiprows=1, dtype=type("t"))
military = np.zeros(shape=mili.shape)
for i in range(0, len(mili)):
    military[i][0] = float(np.char.strip(mili[i][0], '"'))
    military[i][1] = float(np.char.strip(mili[i][1], ' "'))
dataSets = [cal, river, military]
dataLabels = ["California", "River", "Military"]
dataColors = ['red', 'blue', 'green']

sVals = [0.1, 0.5, 2, 4, 10]
neighborOrdinals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
trials = 500
ratesSNeigh = np.zeros(shape=10)
distsSZero = np.zeros(shape=(len(sVals)))
ratesSZero = np.zeros(shape=(len(sVals)))
ratesSNeighSave = np.zeros(shape=(len(dataColors), 10))
distsSZeroSave = np.zeros(shape=(3, len(sVals)))
ratesSZeroSave = np.zeros(shape=(3, len(sVals)))
rate = np.zeros(shape=(3, len(sVals), 10, trials))
dists = np.zeros(shape=(3, len(sVals), 10, trials))
for i in range(0, len(dataSets)):
    print("\n", dataLabels[i], "\n")
    rate[i], dists[i] = fn.stuffSValues(sVals, trials, dataSets[i], 25, 125)
for i in range(0, len(dataSets)):
    for sIndex in range(0, len(sVals)):
        ratesSZero[sIndex] = np.mean(rate[i][sIndex][0])
    ratesSZeroSave[i] = ratesSZero
    plt.plot(sVals, ratesSZero, color=dataColors[i], label=dataLabels[i])
np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\ratesS.txt", X=ratesSZeroSave, delimiter=',')
plt.title("Mean error rate as a function of s value")
plt.gca().set_ylim([0., 0.5])
ax = plt.subplot(111)
ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\ratesS")
plt.clf()
for i in range(0, len(dataSets)):
    for sIndex in range(0, len(sVals)):
        distsSZero[sIndex] = np.mean(dists[i][sIndex][0])
    distsSZeroSave[i] = distsSZero
    plt.plot(sVals, distsSZero, color=dataColors[i], label=dataLabels[i])
np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\distsS.txt", X=distsSZeroSave, delimiter=',')
plt.title("Mean error magnitude as a function of s value")
plt.gca().set_ylim([0., 0.5])
ax = plt.subplot(111)
ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\distsS")
plt.clf()
for sIndex in range(0, len(sVals)):
    for i in range(0, len(dataSets)):
        for neigh in range(0, 10):
            ratesSNeigh[neigh] = np.mean(rate[i][sIndex][neigh])
            ratesSNeighSave[i] = ratesSNeigh
        plt.plot(neighborOrdinals, ratesSNeigh, color=dataColors[i], label=dataLabels[i])
    np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\ratesOrd" + str(sIndex) + ".txt",
               X=ratesSNeighSave, delimiter=',')
    plt.title("Mean error rate versus ordinality for s = " + str(sVals[sIndex]))
    plt.gca().set_ylim([0., 1.])
    ax = plt.subplot(111)
    ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
    plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\rateOrd" + str(sIndex))
    plt.clf()
