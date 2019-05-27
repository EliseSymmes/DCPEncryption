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
    np.random.seed(int(math.fabs(total + s + a)))
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


def plotErrorDistance(s, a, coordSize, dim, trials,
                      logScale=False, exclZeros=True, draw=True, neighborhood=None, numNeighbors=100):
    if neighborhood is None and numNeighbors is not None:
        neighborhood = genSetPoints(dim, numNeighbors, coordSize)
    elif numNeighbors is None:
        print("Exactly one of neighborhood or numNeighbors must be provided")
        return
    encNeighborhood = encryptArr(neighborhood, s, a)
    offBy = np.zeros(shape=trials)
    numDif = 0
    print(len(neighborhood))
    for i in range(0, trials):
        point = genPoint(dim, coordSize)
        encPoint = encryptSingle(point, s, a)
        normNeighbor = nearestNeighbor(point, neighborhood)
        encNeighbor = nearestNeighbor(encPoint, encNeighborhood)
        if normNeighbor != encNeighbor:
            numDif += 1
            offBy[i] = math.fabs(distance(point, neighborhood[normNeighbor]
                                          - distance(point, neighborhood[encNeighbor])))
    if draw and (numDif > 0 or not exclZeros):
        if exclZeros:
            offBy = offBy.take(offBy.nonzero()).reshape(numDif)
        plot.hist(offBy, color='blue', log=logScale)
        plot.suptitle("Error distance ith s = " + str(s) + ", a = " +str(a) + "\n" +
                      str(numDif) + " errors from " + str(trials) + " trials within "
                      + str(coordSize) + " side length cube with " + str(len(neighborhood)) + " neighbors")
        plot.show()
    return float((numDif/trials)*100)


def numberOverlap(neighborhood, alpha):
    numOverlap = 0
    status = np.full(len(neighborhood), False)
    for i in range(0, len(neighborhood)):
        for j in range(i + 1, len(neighborhood)):
            if distance(neighborhood[i], neighborhood[j]) <= alpha / 2:
                if not status[j]:
                    numOverlap += 1
                    status[j] = True
                if not status[i]:
                    numOverlap += 1
                    status[i] = True
    return numOverlap


def scatter2d(x, alpha=0, enc=False):
    if 2 != (len(x[0])):
        print("wrong dimensions", len(x[0]))
        return
    xCoord = np.zeros(shape=len(x))
    yCoord = np.zeros(shape=len(x))
    for i in range(0, len(x)):
        xCoord[i] = x[i][0]
        yCoord[i] = x[i][1]
    boundUp = max(max(xCoord), max(yCoord))
    boundDw = min(min(xCoord), min(yCoord))
    area = ((alpha / (boundUp - boundDw)) * np.full(shape=len(x), fill_value=143)) ** 2
    dotArea = ((alpha / (boundUp - boundDw)) * np.full(shape=len(x), fill_value=10)) ** 2
    plot.axis([boundDw, boundUp*1.25, boundDw, boundUp])
    if alpha != 0:
        plot.scatter(xCoord, yCoord, s=area, c='black', alpha=0.2)
        if enc:
            encXCoord = np.zeros(shape=len(x))
            encYCoord = np.zeros(shape=len(x))
            encSet = encryptArr(x, 1, alpha)
            for i in range(0, len(x)):
                encXCoord[i] = encSet[i][0]
                encYCoord[i] = encSet[i][1]
            plot.scatter(encXCoord, encYCoord, s=dotArea, c='blue', alpha=1)
    plot.scatter(xCoord, yCoord, s=2*dotArea, c='black', alpha=1)
    plot.show()
