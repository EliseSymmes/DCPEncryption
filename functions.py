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


def getNeighbor(query, neighborhood, s, a, sigmaStar, k, ordinal=False):
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


def getNeighbors(query, neighborhood, neighbors):
    ret = np.zeros(shape=(len(query), neighbors))
    for q in range(0, len(query)):
        distArr = np.full(shape=neighbors, fill_value=sys.float_info.max)
        indArr = np.full(shape=neighbors, fill_value=-1)
        for n in range(0, len(neighborhood)):
            dist = distance.euclidean(query[q], neighborhood[n])
            tempIndexNew = n
            for i in range(0, neighbors):
                if distArr[i] > dist:
                    tempDist = distArr[i]
                    distArr[i] = dist
                    dist = tempDist
                    tempIndexOld = indArr[i]
                    indArr[i] = tempIndexNew
                    tempIndexNew = tempIndexOld
        ret[q] = indArr
    return ret


def errorDistance(query, neighborhood, s, a, sigmaStar, k):
    norm, enc, ordinal = getNeighbor(query, neighborhood, s, a, sigmaStar, k, ordinal=True)
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
        preferred = num
        # preferred = noZerosPlease(num)
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


def statsDump(trials, configs, points, querySize, neighborSize, sigma):
    errorRate = np.zeros(shape=trials)
    errorDist = np.zeros(shape=trials)
    numBetter = np.zeros(shape=trials)
    configRet = np.zeros(shape=(configs, 3))
    storage = np.zeros(shape=(configs, 3, trials))
    for config in range(0, configs):
        s, alpha, sigmaStar, k = keygen(sigma)
        configRet[config][0] = s
        configRet[config][1] = alpha
        configRet[config][2] = sigmaStar
        for trial in range(0, trials):
            query, neighbors = splitArr(points, querySize, neighborSize)
            errorDists, num = errorDistance(query, neighbors, s, alpha, sigmaStar, k)
            noZeros = noZerosPlease(errorDists)
            errorRate[trial] = len(noZeros) / len(errorDists)
            errorDist[trial] = np.mean(errorDists)
            numBetter[trial] = np.mean(num)
        storage[config][0] = errorRate
        storage[config][1] = errorDist
        storage[config][2] = numBetter
    return configRet, storage


def statsParams(trials, alphaVals, sVals, sigmaStarVals, points, querySize, neighborSize, neighborsRet, acceptable):
    errorRate = np.zeros(shape=(len(alphaVals), len(sVals), len(sigmaStarVals)))
    k = np.random.randint(50)
    for alpha in range(0, len(alphaVals)):
        for s in range(0, len(sVals)):
            for sigmaStar in range(0, len(sigmaStarVals)):
                trialRate = np.zeros(shape=trials)
                for trial in range(0, trials):
                    query, neighbors = splitArr(points, querySize, neighborSize)
                    normNeighbors = getNeighbors(query, neighbors, neighborsRet)
                    encQuery, nums = encryptArr(query, sVals[s], alphaVals[alpha], sigmaStarVals[sigmaStar], k)
                    encNeighbor, nums = encryptArr(neighbors, sVals[s], alphaVals[alpha], sigmaStarVals[sigmaStar], k)
                    encNeighbors = getNeighbors(encQuery, encNeighbor, neighborsRet)
                    wrong = 0
                    for i in range(0, querySize):
                        if len(np.setdiff1d(normNeighbors[i], encNeighbors[i])) > acceptable:
                            wrong += 1
                    trialRate[trial] = wrong / querySize
                    print("\r" + str(alpha) + " - " + str(s) + " - " + str(sigmaStar) + " - " + str(trial), end='')
                errorRate[alpha][s][sigmaStar] = np.mean(trialRate)
    return errorRate


def stuffSValues(sVals, trials, points, querySize, neighborSize):
    errorRate = np.zeros(shape=(len(sVals), 10, trials))
    errorDists = np.zeros(shape=(len(sVals), 10, trials))
    for sIndex in range(0, len(sVals)):
        for trial in range(0, trials):
            print("\r" + str(sVals[sIndex]) + " - " + str(trial), end='')
            dists = np.zeros(shape=(10, querySize))
            rates = np.zeros(shape=(10, querySize))
            nope, alpha, sigmaStar, k = keygen(10)
            query, neighbors = splitArr(points, querySize, neighborSize)
            normNeighbors = getNeighbors(query, neighbors, 10)
            encQuery, nums = encryptArr(query, sVals[sIndex], alpha, sigmaStar, k)
            encNeighbor, nums = encryptArr(neighbors, sVals[sIndex], alpha, sigmaStar, k)
            encNeighbors = getNeighbors(encQuery, encNeighbor, 10)
            for i in range(0, querySize):
                for j in range(0, 10):
                    if normNeighbors[i][j] != encNeighbors[i][j]:
                        rates[j][i] = 1.
                        dists[j][i] = distance.euclidean(neighbors[int(normNeighbors[i][j])],
                                                         neighbors[int(encNeighbors[i][j])])
            for i in range(0, 10):
                errorRate[sIndex][i][trial] = np.mean(rates[i])
                errorDists[sIndex][i][trial] = np.mean(dists[i])
    return errorRate, errorDists


def stuffAValues(aVals, trials, points, querySize, neighborSize):
    errorRate = np.zeros(shape=(len(aVals), 10, trials))
    errorDists = np.zeros(shape=(len(aVals), 10, trials))
    for aIndex in range(0, len(aVals)):
        for trial in range(0, trials):
            print("\r" + str(aVals[aIndex]) + " - " + str(trial), end='')
            dists = np.zeros(shape=(10, querySize))
            rates = np.zeros(shape=(10, querySize))
            nope, alpha, sigmaStar, k = keygen(10)
            query, neighbors = splitArr(points, querySize, neighborSize)
            normNeighbors = getNeighbors(query, neighbors, 10)
            encQuery, nums = encryptArr(query, 0.5, aVals[aIndex], sigmaStar, k)
            encNeighbor, nums = encryptArr(neighbors, 0.5, aVals[aIndex], sigmaStar, k)
            encNeighbors = getNeighbors(encQuery, encNeighbor, 10)
            for i in range(0, querySize):
                for j in range(0, 10):
                    if normNeighbors[i][j] != encNeighbors[i][j]:
                        rates[j][i] = 1.
                        dists[j][i] = distance.euclidean(neighbors[int(normNeighbors[i][j])],
                                                         neighbors[int(encNeighbors[i][j])])
            for i in range(0, 10):
                errorRate[aIndex][i][trial] = np.mean(rates[i])
                errorDists[aIndex][i][trial] = np.mean(dists[i])
    return errorRate, errorDists


def rateSA(sVals, aVals, points, trials):
    ret = np.zeros(shape=len(sVals))
    for i in range(0, len(sVals)):
        trialSet = np.zeros(shape=trials)
        for trial in range(0, trials):
            print("\r" + str(i) + " - " + str(trial), end='')
            rateNeigh = np.zeros(shape=25)
            _, _, sigmaStar, k = keygen(50)
            query, neighbors = splitArr(points, 25, 125)
            normNeighbors = getNeighbors(query, neighbors, 1)
            encQuery, nums = encryptArr(query, sVals[i], aVals[i], sigmaStar, k)
            encNeighbor, nums = encryptArr(neighbors, sVals[i], aVals[i], sigmaStar, k)
            encNeighbors = getNeighbors(encQuery, encNeighbor, 1)
            for j in range(0, 25):
                if normNeighbors[j] != encNeighbors[j]:
                    rateNeigh[j] = 1.
            trialSet[trial] = np.mean(rateNeigh)
        ret[i] = np.mean(trialSet)
    return ret


def knnRates(points, trials, numNeigh):
    storage = np.zeros(shape=(numNeigh, trials, 25))
    for trial in range(0, trials):
        print("\r" + str(trial), end='')
        s, alpha, sigmaStar, k = keygen(50)
        query, neighbors = splitArr(points, 25, 125)
        normNeighbors = getNeighbors(query, neighbors, numNeigh)
        encQuery, nums = encryptArr(query, s, alpha, sigmaStar, k)
        encNeighbor, nums = encryptArr(neighbors, s, alpha, sigmaStar, k)
        encNeighbors = getNeighbors(encQuery, encNeighbor, numNeigh)
        for q in range(0, 25):
            for i in range(0, numNeigh):
                tmp = np.zeros(shape=i+1)
                for j in range(0, i+1):
                    k = 0
                    while k <= i:
                        if normNeighbors[q][j] != encNeighbors[q][k]:
                            tmp[j] = 1
                            k += 1
                        else:
                            tmp[j] = 0
                            k = i+1
                storage[i][trial][q] = np.mean(tmp)
    ret = np.zeros(shape=numNeigh)
    for i in range(0, numNeigh):
        ret[i] = np.mean(storage[i])
    return ret
