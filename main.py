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

neighVals = fn.makeArr(1, 20)
for i in range(0, len(dataSets)):
    rates = fn.knnRates(dataSets[i], 100, 20)
    plt.plot(neighVals, rates, color=dataColors[i], label=dataLabels[i])
np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\numNeigh\\" + str(dataLabels[i]), X=rates, delimiter=',')
plt.title("Mean error rate for the k nearest neighbors")
plt.gca().set_ylim([0., 1.])
ax = plt.subplot(111)
ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\numNeigh\\neighbors")
plt.clf()

# aVals = [0.1, 0.3, 0.5, 1, 2, 0.1, 0.3, 0.5, 1, 2, 0.5, 1, 2, 2.5, 4, 1, 2, 5, 6, 10, 1, 2, 5, 10, 12]
# sVals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10]
# ratesRiddhi = fn.rateSA(sVals, aVals, military, 300)
# print(ratesRiddhi)
# np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\riddhi.txt", X=ratesRiddhi, delimiter=',')

# aVals = [0.1, 0.2, 0.5, 1.]
# neighborOrdinals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# trials = 800
# ratesANeigh = np.zeros(shape=10)
# distsAZero = np.zeros(shape=(len(aVals)))
# ratesAZero = np.zeros(shape=(len(aVals)))
# ratesANeighSave = np.zeros(shape=(len(dataColors), 10))
# distsAZeroSave = np.zeros(shape=(3, len(aVals)))
# ratesAZeroSave = np.zeros(shape=(3, len(aVals)))
# rate = np.zeros(shape=(3, len(aVals), 10, trials))
# dists = np.zeros(shape=(3, len(aVals), 10, trials))
# for i in range(0, len(dataSets)):
#     print("\n", dataLabels[i], "\n")
#     rate[i], dists[i] = fn.stuffAValues(aVals, trials, dataSets[i], 25, 125)
# for i in range(0, len(dataSets)):
#     for aIndex in range(0, len(aVals)):
#         ratesAZero[aIndex] = np.mean(rate[i][aIndex][0])
#     ratesAZeroSave[i] = ratesAZero
#     plt.plot(aVals, ratesAZero, color=dataColors[i], label=dataLabels[i])
# np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\alpha\\ratesA.txt", X=ratesAZeroSave, delimiter=',')
# plt.title("Mean error rate as a function of alpha value")
# plt.gca().set_ylim([0., 0.5])
# ax = plt.subplot(111)
# ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
# plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\alpha\\ratesA")
# plt.clf()
# for i in range(0, len(dataSets)):
#     for aIndex in range(0, len(aVals)):
#         distsAZero[aIndex] = np.mean(dists[i][aIndex][0])
#     distsAZeroSave[i] = distsAZero
#     plt.plot(aVals, distsAZero, color=dataColors[i], label=dataLabels[i])
# np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\alpha\\distsA.txt", X=distsAZeroSave, delimiter=',')
# plt.title("Mean error magnitude as a function of alpha value")
# plt.gca().set_ylim([0., 0.3])
# ax = plt.subplot(111)
# ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
# plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\alpha\\distsA")
# plt.clf()
# for aIndex in range(0, len(aVals)):
#     for i in range(0, len(dataSets)):
#         for neigh in range(0, 10):
#             ratesANeigh[neigh] = np.mean(rate[i][aIndex][neigh])
#             ratesANeighSave[i] = ratesANeigh
#         plt.plot(neighborOrdinals, ratesANeigh, color=dataColors[i], label=dataLabels[i])
#     np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\alpha\\ratesOrd" + str(aIndex) + ".txt",
#                X=ratesANeighSave, delimiter=',')
#     plt.title("Mean error rate versus ordinality for alpha = " + str(aVals[aIndex]))
#     plt.gca().set_ylim([0., 1.])
#     ax = plt.subplot(111)
#     ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
#     plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\alpha\\ratesOrd" + str(aIndex))
#     plt.clf()


# for i in range(0, len(dataSets)):
#     print("\n", dataLabels[i], "\n")
#     rate[i], dists[i] = fn.stuffSValues(sVals, trials, dataSets[i], 25, 125)
# for i in range(0, len(dataSets)):
#     for sIndex in range(0, len(sVals)):
#         ratesSZero[sIndex] = np.mean(rate[i][sIndex][0])
#     ratesSZeroSave[i] = ratesSZero
#     plt.plot(sVals, ratesSZero, color=dataColors[i], label=dataLabels[i])
# np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\ratesS.txt", X=ratesSZeroSave, delimiter=',')
# plt.title("Mean error rate as a function of s value")
# plt.gca().set_ylim([0., 0.3])
# ax = plt.subplot(111)
# ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
# plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\ratesS")
# plt.clf()
# for i in range(0, len(dataSets)):
#     for sIndex in range(0, len(sVals)):
#         distsSZero[sIndex] = np.mean(dists[i][sIndex][0])
#     distsSZeroSave[i] = distsSZero
#     plt.plot(sVals, distsSZero, color=dataColors[i], label=dataLabels[i])
# np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\distsS.txt", X=distsSZeroSave, delimiter=',')
# plt.title("Mean error magnitude as a function of s value")
# plt.gca().set_ylim([0., 0.2])
# ax = plt.subplot(111)
# ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
# plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\distsS")
# plt.clf()
# for sIndex in range(0, len(sVals)):
#     for i in range(0, len(dataSets)):
#         for neigh in range(0, 10):
#             ratesSNeigh[neigh] = np.mean(rate[i][sIndex][neigh])
#             ratesSNeighSave[i] = ratesSNeigh
#         plt.plot(neighborOrdinals, ratesSNeigh, color=dataColors[i], label=dataLabels[i])
#     np.savetxt(fname="C:\\Users\\Elise\\Pictures\\charts\\ratesOrd" + str(sIndex) + ".txt",
#                X=ratesSNeighSave, delimiter=',')
#     plt.title("Mean error rate versus ordinality for s = " + str(sVals[sIndex]))
#     plt.gca().set_ylim([0., 1.])
#     ax = plt.subplot(111)
#     ax.legend(loc='upper right', bbox_to_anchor=(0.5, 1.0), shadow=True, ncol=2)
#     plt.savefig("C:\\Users\\Elise\\Pictures\\charts\\ratesOrd" + str(sIndex))
#     plt.clf()
