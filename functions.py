import random
import numpy as np
import sys
import math


def genPoint(d, coordLowerBound, coordUpperBound):
    x = np.zeros(shape=d)
    for i in range(0, d):
        x[i] = random.uniform(coordLowerBound, coordUpperBound)
    return x


def genSetPoints(d, num, coordLowerBound, coordUpperBound):
    ret = np.zeros(shape=(num, d))
    for i in range(0, num):
        ret[i] = genPoint(d, coordLowerBound, coordUpperBound)
    return ret


def p(pnt, s, a):
    total = 0.0
    for i in pnt:
        total += i
    random.seed(total + s + a)
    normal = random.uniform(0, math.fabs(s*a/4))
    weights = np.random.rand(len(pnt))
    normal *= normal
    weightsTotal = 0.0
    for i in weights:
        weightsTotal += i
    ret = np.zeros(shape=pnt.shape)
    for i in range(0, len(ret)):
        ret[i] = math.sqrt(normal*(weights[i]/weightsTotal))
        if random.randrange(0, 2) == 1:
            ret[i] *= -1
    return ret


def encryptSingle(x, s, a):
    ret = np.zeros(shape=x.shape)
    pVals = p(x, s, a)
    for i in range(0, len(x)):
        ret[i] = x[i] * s + pVals[i]
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
