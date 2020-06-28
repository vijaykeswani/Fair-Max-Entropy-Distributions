import numpy as np

from FairMaxEnt.maximum_entropy_distribution import MaxEnt
from FairMaxEnt.memory import MemoryTrie


def reweightSamples(samples, weights, Xindices, Yindices):
    XYweights = {}
    Yweights = {}
    output = np.zeros_like(weights, dtype=float)
    for weight, sample in zip(weights, samples):
        key1 = tuple(sample[Yindices])
        key2 = tuple(sample[Xindices])
        XYweights[key1] = XYweights.get(key1, {})
        XYweights[key1][key2] = XYweights[key1].get(key2, 0) + weight
        Yweights[key1] = Yweights.get(key1, 0) + weight

    for i, (weight, sample) in enumerate(zip(weights, samples)):
        key1 = tuple(sample[Yindices])
        key2 = tuple(sample[Xindices])
        XYweight = XYweights[key1][key2]
        Yweight = Yweights[key1]
        output[i] = weight / XYweight * Yweight / sum(Yweights.values())
    return output


def computeMean(domain, samples, weights, newWeights, weightedMean, alterMean, index):
    if weightedMean:
        mean = np.dot(newWeights, samples)/np.sum(newWeights)
    else:
        mean = np.dot(weights, samples)/np.sum(weights)
    if alterMean:
        dimensions = [domain.dimensionOfFeature(None, i) for i in range(index+1)]
        start = sum(dimensions[:index])
        end = start + dimensions[index]
        uniqueValues = domain.getUniqueValues(None, index)
        newMean = np.mean(uniqueValues, axis=0)
        mean[start:end] = newMean
    return mean


def fitMaxEnt(maxEnt, samples, weights, C, mean, epsilon, coordinate, lower, upper, bestValue=-float("inf")):
    if upper - lower < epsilon:
        mean[coordinate] = (upper+lower)/2
        lambda_, value = maxEnt.fit(samples, weights, C, mean, earlyCut=bestValue, eps=epsilon)
        return lambda_

    high = (lower + 3*upper)/4
    mid  = (lower + upper)/2
    low  = (3*lower + upper)/4

    mean[coordinate] = low
    _, valueLow = maxEnt.fit(samples, weights, C, mean, earlyCut=bestValue, eps=epsilon)
    bestValue = max(valueLow, bestValue)

    mean[coordinate] = mid
    _, valueMid = maxEnt.fit(samples, weights, C, mean, earlyCut=bestValue, eps=epsilon)
    bestValue = max(valueMid, bestValue)

    mean[coordinate] = high
    _, valueHigh = maxEnt.fit(samples, weights, C, mean, earlyCut=bestValue, eps=epsilon)
    bestValue = max(valueHigh, bestValue)

    if valueLow >=  max(valueMid, valueHigh):
        lower, upper = lower, mid
    elif valueHigh > valueMid:
        lower, upper = mid, upper
    else:
        lower, upper = low, high
    return fitMaxEnt(maxEnt, samples, weights, C, mean, epsilon, coordinate, lower, upper, bestValue)


def FairMaximumEntropy(domain, samples, C, delta, attributeWithMarginIndex,
                         reweight=False, reweightXindices = tuple(),
                         reweightYindices = tuple(),
                         weightedMean=False, alterMean=False,
                         priorMemory=MemoryTrie, probabilityMemory=MemoryTrie,
                         epsilon=1e-8):
    maxEnt = MaxEnt(domain, priorMemory(), probabilityMemory())
    samples, weights = domain.compress(samples)

    if reweight:
        newWeights = reweightSamples(samples, weights, reweightXindices, reweightYindices)
    else:
        newWeights = weights
    if weightedMean:
        newWeights2 = reweightSamples(samples, weights, reweightXindices, reweightYindices)
    else:
        newWeights2 = weights
       
        
#     coordinate = sum(domain.dimensionOfFeature(None, i) for i in range(attributeWithMarginIndex))
    mean = computeMean(domain, samples, weights, newWeights2, weightedMean, alterMean, attributeWithMarginIndex)

#     print("Mean: ", mean)
#     if delta != 0:
#         lambda_ = fitMaxEnt(maxEnt, samples, newWeights, C, mean, epsilon, coordinate, mean[coordinate]-delta, mean[coordinate]+delta)
#     else:
    
    lambda_, value = maxEnt.fit(samples, newWeights, C, mean, earlyCut=-float("inf"), eps=epsilon)
    maxEnt.initialize_for_sampling(lambda_, samples, newWeights, C)
    return maxEnt

