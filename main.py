import random
import numpy as np
import matplotlib.pyplot as plot
import functions as fn

s = random.uniform(-2.0, 2.0)
a = random.uniform(-1.0, 1.0)
coordLowerBound = -100.0
coordUpperBound = 100.0
dim = 3
numNeighbors = 1000
trials = 10000

neighborhood = fn.genSetPoints(dim, numNeighbors, coordLowerBound, coordUpperBound)
encNeighborhood = fn.encryptArr(neighborhood, s, a)
offBy = np.zeros(shape=trials)
offByEnc = np.zeros(shape=trials)
numSame = 0
numDif = 0
for i in range(0, trials):
    point = fn.genPoint(dim, coordLowerBound, coordUpperBound)
    encPoint = fn.encryptSingle(point, s, a)
    normNeighbor = fn.nearestNeighbor(point, neighborhood)
    encNeighbor = fn.nearestNeighbor(encPoint, encNeighborhood)
    if normNeighbor == encNeighbor:
        numSame += 1
    else:
        numDif += 1
        offBy[i] = fn.distance(neighborhood[normNeighbor], neighborhood[encNeighbor])
        offByEnc[i] = fn.distance(encNeighborhood[normNeighbor], encNeighborhood[encNeighbor])
offBy = offBy.take(offBy.nonzero()).reshape(numDif)
offByEnc = offByEnc.take(offByEnc.nonzero()).reshape(numDif)
print(numDif, "errors from", trials, "trials")
plot.hist(offByEnc)
plot.show()
