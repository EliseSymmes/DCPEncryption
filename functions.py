import numpy as np
import sys
import math
import matplotlib.pyplot as plot
import scipy.spatial.distance as distance


def makeArr(step, iterations):
    ret = np.array([step])
    for i in range(1, iterations):
        ret = np.append(ret, step+(step*i))
    return ret


def keygen(sigma):
    np.random.seed()
    alpha = np.random.uniform(0.2, 1.0)
    sigmaStar = 0.0
    while sigmaStar <= 0.0:
        sigmaStar = np.random.normal(0.0, math.sqrt(sigma))
    s = 0.0
    while s <= 0.0:
        s = np.random.normal(0.0, math.sqrt(sigmaStar))
    k = np.random.randint(2147483647)
    return s, alpha, sigmaStar, k


def genPoint(d, lower, upper):
    x = np.zeros(shape=d)
    for i in range(0, d):
        x[i] = np.random.uniform(lower, upper)
    return x


def genSetPoints(d, num, lower, upper):
    ret = np.zeros(shape=(num, d))
    for i in range(0, num):
        ret[i] = genPoint(d, lower, upper)
    return ret


def badHash(num, k):
    np.random.seed((num * k) % 2147483648)
    return np.random.randint(2147483647)


def p(pnt, s, a, sigmaStar, randNum, k):
    ret = np.full(shape=pnt.shape, fill_value=sys.float_info.max)
    np.random.seed(badHash(randNum, k))
    bound = s * a / (math.sqrt(len(pnt)) * 4)
    for i in range(0, len(ret)):
        while math.fabs(ret[i]) > bound:
            ret[i] = np.random.normal(0.0, math.sqrt(sigmaStar))
    return ret


def encryptSingle(x, s, a, sigmaStar, k):
    ret = np.zeros(shape=x.shape)
    rand = np.random.randint(2147483647)
    pVal = p(x, s, a, sigmaStar, rand, k)
    for i in range(0, len(x)):
        ret[i] = x[i] * s + pVal[i]
    return ret, rand


def encryptArr(x, s, a, sigmaStar, k):
    encX = np.zeros(shape=x.shape)
    nums = np.array(0, dtype=type(1))
    for i in range(0, len(x)):
        encX[i], num = encryptSingle(x[i], s, a, sigmaStar, k)
        nums = np.append(nums, num)
    return encX, np.delete(nums, [0], axis=0)


def nearestNeighbor(point, neighbors):
    nearestDist = sys.float_info.max
    nearestIndex = -1
    for n in range(0, len(neighbors)):
        dist = distance.euclidean(point, neighbors[n])
        if dist < nearestDist:
            nearestIndex = n
            nearestDist = dist
    return nearestIndex


def nearestNeighborOrd(point, neighbors):
    nearestDist = sys.float_info.max
    nearestIndex = -1
    distances = np.zeros(shape=len(neighbors))
    for n in range(0, len(neighbors)):
        dist = distance.euclidean(point, neighbors[n])
        distances[n] = dist
        if dist < nearestDist:
            nearestIndex = n
            nearestDist = dist
    return nearestIndex, distances


def plotPDist(num):
    portions = np.zeros(shape=num)
    for i in range(0, num):
        s, a, sigmaStar = keygen(50)
        x = genPoint(2, 0, 15)
        pVal = p(x, s, a, sigmaStar, i)
        portions[i] = math.fabs((np.linalg.norm(pVal) - math.fabs(s * a / 4)) / (s * a / 4))
    plot.hist(portions)
    plot.suptitle("Distribution of p values divided by sa/4")
    plot.show()


def getNeighbors(query, neighborhood, s, a, sigmaStar, k, ordinal=False):
    normNeighbors = np.zeros(shape=len(query))
    encNeighbors = np.zeros(shape=len(query))
    offBy = np.zeros(shape=(len(query), len(neighborhood)))
    encQuery, nums = encryptArr(query, s, a, sigmaStar, k)
    encNeighborhood, nums = encryptArr(neighborhood, s, a, sigmaStar, k)
    for i in range(0, len(query)):
        normNeighbors[i],  offBy[i] = nearestNeighborOrd(query[i], neighborhood)
        encNeighbors[i] = nearestNeighbor(encQuery[i], encNeighborhood)
    if ordinal:
        return normNeighbors, encNeighbors, offBy
    else:
        return normNeighbors, encNeighbors


def errorDistance(query, neighborhood, s, a, sigmaStar, k):
    norm, enc, ordinal = getNeighbors(query, neighborhood, s, a, sigmaStar, k, ordinal=True)
    errorDist = np.zeros(shape=len(query))
    numBetter = np.zeros(shape=len(query))
    for i in range(0, len(query)):
        if norm[i] != enc[i]:
            errorDist[i] = distance.euclidean(neighborhood[int(norm[i])], neighborhood[int(enc[i])])
            for j in range(0, len(ordinal[i])):
                if ordinal[i][j] < ordinal[i][int(enc[i])]:
                    numBetter[i] += 1
    return errorDist, numBetter


def noZerosPlease(values):
    j = 0
    for i in range(0, len(values)):
        if values[j] == 0.0:
            values = np.delete(values, j, 0)
        else:
            j += 1
    return values


def numberOverlap(neighborhood, alpha):
    numOverlap = 0
    status = np.full(len(neighborhood), False)
    for i in range(0, len(neighborhood)):
        for j in range(i + 1, len(neighborhood)):
            if distance.euclidean(neighborhood[i], neighborhood[j]) <= alpha / 2:
                if not status[j]:
                    numOverlap += 1
                    status[j] = True
                if not status[i]:
                    numOverlap += 1
                    status[i] = True
    return numOverlap


def scatter2d(x, sigma=50, enc=False):
    if 2 != (len(x[0])):
        print("wrong dimensions", len(x[0]))
        return
    s, alpha, sigmaStar = keygen(sigma)
    xCoord = np.zeros(shape=len(x))
    yCoord = np.zeros(shape=len(x))
    for i in range(0, len(x)):
        xCoord[i] = x[i][0]
        yCoord[i] = x[i][1]
    boundUp = max(max(xCoord), max(yCoord))
    boundDw = min(min(xCoord), min(yCoord))
    area = ((alpha / (boundUp - boundDw)) * np.full(shape=len(x), fill_value=143)) ** 2
    dotArea = ((alpha / (boundUp - boundDw)) * np.full(shape=len(x), fill_value=10)) ** 2
    plot.axis([boundDw, boundUp * 1.25, boundDw, boundUp])
    if alpha != 0:
        plot.scatter(xCoord, yCoord, s=area, c='black', alpha=0.2)
        if enc:
            encXCoord = np.zeros(shape=len(x))
            encYCoord = np.zeros(shape=len(x))
            encSet, nums = encryptArr(x, 1, alpha, sigmaStar, 10)
            for i in range(0, len(x)):
                encXCoord[i] = encSet[i][0]
                encYCoord[i] = encSet[i][1]
            plot.scatter(encXCoord, encYCoord, s=dotArea, c='blue', alpha=1)
    plot.scatter(xCoord, yCoord, s=2 * dotArea, c='black', alpha=1)
    plot.show()


def splitArr(arr, sample1, sample2, replacement=False):
    if sample1 + sample2 > len(arr) and not replacement:
        print("Invalid sizing")
        return
    ret1 = np.zeros(shape=(sample1, len(arr[0])))
    ret2 = np.zeros(shape=(sample2, len(arr[0])))
    for i in range(0, sample1):
        k = int(math.trunc(np.random.uniform(0, len(arr))))
        ret1[i] = arr[k]
        if replacement:
            arr = np.delete(arr, k, 0)
    for i in range(0, sample2):
        k = int(math.trunc(np.random.uniform(0, len(arr))))
        ret2[i] = arr[k]
        if not replacement:
            arr = np.delete(arr, k, 0)
    return ret1, ret2


def statsTrials(trials, sigma, query, neighbors):
    errorRate = np.zeros(shape=trials)
    errorDist = np.zeros(shape=trials)
    for trial in range(0, trials):
        s, a, sigmaStar, k = keygen(sigma)
        errorDists, numBetter = errorDistance(query, neighbors, s, a, sigmaStar, k)
        noZeros = noZerosPlease(errorDists)
        errorRate[trial] = len(noZeros) / len(errorDists)
        errorDist[trial] = np.mean(errorDists)
        print("\rTrial " + str(trial+1) + " with sigma " + str(sigma) + " complete", end='')
    return errorRate, errorDist


def statsTrialsSplit(trials, sigma, points, querySize, neighborSize):
    errorRate = np.zeros(shape=trials)
    errorDist = np.zeros(shape=trials)
    numBetter = np.zeros(shape=trials)
    for trial in range(0, trials):
        query, neighbors = splitArr(points, querySize, neighborSize)
        s, a, sigmaStar, k = keygen(sigma)
        errorDists, num = errorDistance(query, neighbors, s, a, sigmaStar, k)
        noZeros = noZerosPlease(errorDists)
        errorRate[trial] = len(noZeros) / len(errorDists)
        errorDist[trial] = np.mean(errorDists)
        preferred = noZerosPlease(num)
        if len(preferred) > 0:
            numBetter[trial] = np.mean(preferred)
        print("\rTrial " + str(trial+1) + " with neighbors " + str(neighborSize) + " complete", end='')
    return errorRate, errorDist, numBetter


def statsSigmaRange(points, querySize, neighborsSize, sigmaLower, sigmaUpper, trials, step):
    sigmaRange = sigmaUpper-sigmaLower
    distSigma = np.zeros(shape=1+int(math.trunc(sigmaRange / step)))
    rateSigma = np.zeros(shape=1+int(math.trunc(sigmaRange / step)))
    sigma = sigmaLower
    i = 0
    query, neighbors = splitArr(points, querySize, neighborsSize)
    while sigma <= sigmaUpper:
        errorRate, meanError = statsTrials(trials, sigma, query, neighbors)
        distSigma[i] = np.mean(meanError)
        rateSigma[i] = np.mean(errorRate)
        print("\nSigma =", sigma, "\nMean Error Rate:", rateSigma[i],
              "\nMean Error Distance:", distSigma[i])
        sigma += step
        i += 1
    return rateSigma, distSigma


def statsNeighborRange(points, sigma, querySize, neighborSizes, trials):
    rates = np.zeros(shape=(len(neighborSizes), trials))
    dists = np.zeros(shape=(len(neighborSizes), trials))
    numBetter = np.zeros(shape=(len(neighborSizes), trials))
    for i in range(0, len(neighborSizes)):
        rates[i], dists[i], numBetter[i] = statsTrialsSplit(trials, sigma, points, querySize, neighborSizes[i])
        print("\n", neighborSizes[i], "complete")
    return rates, dists, numBetter
