import numpy as np

from FairMaxEnt.box_constrained_newton_method import BoxConstrainedNewtonMethod as BCNM
from FairMaxEnt.counting_oracle import prepareOracleForBCNM, prepareMarginalProbabilities


class MaxEnt(object):
    def __init__(self, domain, priorMemory, probabilityMemory):
        """
            domain is an instance of FairMaxEnt.domain.Domain
            priorMemory should be a memory model from FairMaxEnt.memory that stores given data
            probabilityMemory should be a memory model from FairMaxEnt.memory
        """
        self.priorMemory = priorMemory
        self.probabilityMemory = probabilityMemory
        self.domain = domain
        self.lambda_ = np.zeros(domain.dimension)
        self.countingOracle = prepareOracleForBCNM(domain)
        self.marginalProbability = prepareMarginalProbabilities(domain)


    def reset(self):
        self.lambda_ = np.zeros(self.domain.dimension)
        self.priorMemory.reset()
        self.probabilityMemory.reset()


    def fit(self, points, weights, C, theta, robustnessCoefficient=4, R=700, eps=1e-8, **kwargs):
        """ kwargs passed to box constrained newton method
            R is the bound on the size of epsilon approximate solution
            700 is approximately double range for exp(x)
        """
        lambda_ = self.lambda_
        alpha = eps / (self.domain.dimension * R **2)
        self.countingOracle.set(Samples=points,
                                Prior=weights,
                                C=C,
                                Alpha=alpha,
                                Mean=theta)

        lambda_, dualValue, info = BCNM(lambda_, self.countingOracle, robustnessCoefficient, R, eps, **kwargs)
        self.lambda_ = lambda_
        return lambda_, dualValue

    def initialize_for_sampling(self, lambda_, points, weights, C):
        self.lambda_ = lambda_
        self.marginalProbability.set(Samples=points, Prior=weights, C=C, Lambda=lambda_)

        encodings = self.domain.toIndices(points)

        for point, encoding, weight in zip(points, encodings, weights):
            value = weight * np.exp(np.dot(point, lambda_))
            for index in range(self.domain.numberOfFeatures, -1, -1):
                encoding[index:] = -1
                self.priorMemory[encoding+1] += value



    def predict_proba_indices(self, indices):
        if indices+1 in self.probabilityMemory:
            output = self.probabilityMemory[indices+1]
        else:
            output = self.marginalProbability(self.priorMemory[indices+1], indices)
            self.probabilityMemory[indices+1] = output
        return output


    def predict_proba(self, data):
        if len(data.shape) == 1:
            data = data.reshape(-1, data.size)
        indices = self.domain.toIndices(data)
        output = np.array(self.predict_proba_indices(index) for index in indices)
        return output

    def _sample(self, mask, p):
        mask = np.array(mask)
        featureIndex = 0
        numFeatures = self.domain.numberOfFeatures
        featureSizes = self.domain.getSizes()

        while True:
            while featureIndex < numFeatures and mask[featureIndex] != -1:
                featureIndex += 1
            if featureIndex == numFeatures:
                break
            featureSize = featureSizes[featureIndex]
            mask[featureIndex] = 0
            q = self.predict_proba_indices(mask)
            while q < p:
                p -= q
                mask[featureIndex] += 1
                if mask[featureIndex] == featureSize:
                    q = p
                    while featureIndex < numFeatures:
                        mask[featureIndex] = featureSizes[featureIndex] - 1
                        featureIndex += 1
                    break
                q = self.predict_proba_indices(mask)

        return mask

    def sample(self, numberOfSamples):
#        if mask is None:
        mask = -np.ones(self.domain.numberOfFeatures, dtype=int)
#        else:
#            mask = mask.flatten().astype(int)
        prior = self.predict_proba_indices(mask)
        P = np.random.uniform(0, prior, numberOfSamples)
        sampleMasks = np.array([self._sample(mask, p) for p in P])

        return self.domain.fromIndices(sampleMasks)

