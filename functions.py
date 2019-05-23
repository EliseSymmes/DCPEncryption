import random
import numpy as np
import sys
import math
import matplotlib.pyplot as plot


def genPoint(d, coordSize):
    x = np.zeros(shape=d)
    for i in range(0, d):
        x[i] = random.uniform(-1 * coordSize / 2, coordSize / 2)
    return x


def genSetPoints(d, num, coordSize):
    ret = np.zeros(shape=(num, d))
    for i in range(0, num):
        ret[i] = genPoint(d, coordSize)
    return ret


def p(pnt, s, a):
    total = 0.0
    for i in pnt:
        total += i
    random.seed(total + s + a)
    normal = random.uniform(0, math.fabs(s * a / 4))
    normal *= normal
    weights = np.random.rand(len(pnt))
    weightsTotal = 0.0
    for i in weights:
        weightsTotal += i
    ret = np.zeros(shape=pnt.shape)
    for i in range(0, len(ret)):
        ret[i] = math.sqrt(normal * (weights[i] / weightsTotal))
        if random.randrange(0, 2) == 1:
            ret[i] *= -1
    return ret


def encryptSingle(x, s, a):
    ret = np.zeros(shape=x.shape)
    pVal = p(x, s, a)
    for i in range(0, len(x)):
        ret[i] = x[i] * s + pVal[i]
    return ret


def encryptArr(x, s, a):
    encX = np.zeros(shape=x.shape)
    for i in range(0, len(x)):
        encX[i] = encryptSingle(x[i], s, a)
    return encX


def distance(pointA, pointB):
    if len(pointA) != len(pointB):
        return
    dist = 0.0
    for i in range(len(pointA)):
        dimDif = pointA[i] - pointB[i]
        dimDif *= dimDif
        dist += dimDif
    return math.sqrt(dist)


def nearestNeighbor(point, neighbors):
    nearestDist = sys.float_info.max
    nearestIndex = -1
    for n in range(0, len(neighbors)):
        dist = distance(point, neighbors[n])
        if dist < nearestDist:
            nearestIndex = n
            nearestDist = dist
    return nearestIndex


def plotPDist(num):
    portions = np.zeros(shape=num)
    for i in range(0, num):
        s = random.uniform(-2.0, 2.0)
        a = random.uniform(-1.0, 1.0)
        x = genPoint(3, 10)
        pVal = p(x, s, a)
        portions[i] = math.fabs((np.linalg.norm(pVal) - math.fabs(s * a / 4)) / (s * a / 4))
    plot.hist(portions)
    plot.suptitle("Distribution of p values divided by sa/4")
    plot.show()


def plotErrorDistance(s, a, coordSize, dim, numNeighbors, trials):
    neighborhood = genSetPoints(dim, numNeighbors, coordSize)
    encNeighborhood = encryptArr(neighborhood, s, a)
    offBy = np.zeros(shape=trials)
    numDif = 0
    for i in range(0, trials):
        point = genPoint(dim, coordSize)
        encPoint = encryptSingle(point, s, a)
        normNeighbor = nearestNeighbor(point, neighborhood)
        encNeighbor = nearestNeighbor(encPoint, encNeighborhood)
        if normNeighbor != encNeighbor:
            numDif += 1
            offBy[i] = math.fabs(distance(point, neighborhood[normNeighbor]
                                          - distance(point, neighborhood[encNeighbor])))
    if numDif > 0:
        offBy = offBy.take(offBy.nonzero()).reshape(numDif)
        plot.hist(offBy, color='blue')
        plot.suptitle("Distance from correct neighbor to selected neighbor\n" +
                      str(numDif) + " errors from " + str(trials) + " trials within "
                      + str(coordSize) + " side length cube")
        plot.show()
    return numDif
