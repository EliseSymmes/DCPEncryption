import random
import numpy as np
import matplotlib.pyplot as plot
import functions as fn
import math

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
        offBy[i] = math.fabs(fn.distance(point, neighborhood[normNeighbor]
                                         - fn.distance(point, neighborhood[encNeighbor])))
        offByEnc[i] = -1 * math.fabs(fn.distance(point, encNeighborhood[normNeighbor]
                                                 - fn.distance(point, encNeighborhood[encNeighbor])))
print("Error rate of", str(float(numDif)*100.0/trials) + "%")
if numDif > 0:
    offBy = offBy.take(offBy.nonzero()).reshape(numDif)
    plot.hist(offBy, color='blue')
    plot.suptitle("Distance from correct neighbor to selected neighbor\n" +
                  str(numDif) + " errors from " + str(trials) + " trials within "
                  + str(coordUpperBound - coordLowerBound) + " side length cube")
    plot.show()
