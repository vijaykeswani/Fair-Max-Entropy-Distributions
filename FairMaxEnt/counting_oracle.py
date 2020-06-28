import os
import pathlib
import pickle

import numpy as np
import theano
import theano.tensor as T
from theano import shared

from Codes.Utils import hash

modelFolder = os.path.join(os.getcwd(), "Models")
pathlib.Path(modelFolder).mkdir(parents=True, exist_ok=True)


theano.config.compute_test_value = 'warn'
theano.config.print_test_value = True
theano.config.exception_verbosity="high"

# Probability model
# given prior q and lambda
# r(x) := q(x)exp(<x,lambda>) / sum_y q(y) exp(<y, lambda>)
# Prior model
# p(x):= count of x, given explicitly
# q(x) = \prod_i q_i(x_i)
# q_i(x_i=v_j) = q_{ij}, given explicitly
# r(x) = (1-C)p(x) + C q(x)
# C is a given constant between 0 and 1


# For construction requires 
# feature descriptions (type, unique_values)
#     supported types
#     categorical
#     binary
#     numerical
# Samples

# Naming convention
# if variable name starts with upper case letter, then it is symbolic
# if variable name starts with lower case letter, then it is python variable
# if function name starts with capital letter, then it returns symbolic output
# if function name starts with capital letter, then it returns python output

def PrepareProbabilities(Support, Prior, Lambda):
    if Prior is None:
        return T.exp(T.dot(Support, Lambda))
    return Prior*T.exp(T.dot(Support, Lambda))

def GetMarginal(Probabilities):
    return T.sum(Probabilities)

def PrepareMarginal(Support, Prior, Lambda):
    # this helper function takes 
    #    - design matrix (the support of the distribution) of size N x n where
    # N is number of points in the support and n is number of features
    #    - prior distribution q(x) (either vector of size N or a scalar in case of uniform prior)
    #    - lambda is the input to the dual program, vector of size n
    # returns three quantities
    #    - \sum_x q(x)e^{<x,lambda>}
    
    # \sum_x q(x)e^{<x,\lambda>}
    return GetMarginal(PrepareProbabilities(Support, Prior, Lambda))


def GetMarginalP0(Prior):
    return GetMarginal(Prior) 

def GetMarginalQ0(domain):
    return shared(domain.size)

def GetQij(domain, Lambda, Support, Mask):
    return PrepareProbabilities(Support, None, Lambda) * Mask

def GetLogMarginalsQ(Qijs):
    return T.log(T.sum(Qijs, axis=1))


def PrepareMarginalP(Samples, Prior, Lambda):
    return PrepareMarginal(Samples, Prior, Lambda)

def PrepareMarginalQ(domain, Lambda, Support, Mask):
    Qijs = GetQij(domain, Lambda, Support, Mask)
    #Qijs = theano.printing.Print("Qijs")(Qijs)
    LogQi = GetLogMarginalsQ(Qijs)
    #LogQi = theano.printing.Print("LogQi")(LogQi)
    LogQ = T.sum(LogQi)
    return T.exp(LogQ)

def JoinPQ(MarginalP, TotalP, MarginalQ, TotalQ, C):
    P = MarginalP / TotalP
    Q = MarginalQ / TotalQ
    return (1-C) * P + C * Q

def PrepareCountingOracle(domain, Samples, Prior, Lambda, C, Support, Mask):
    MarginalP = PrepareMarginalP(Samples, Prior, Lambda)
    TotalP = GetMarginalP0(Prior)
    MarginalQ = PrepareMarginalQ(domain, Lambda, Support, Mask)
    TotalQ = GetMarginalQ0(domain)
    
    #MarginalP = theano.printing.Print("MarginalP")(MarginalP)
    #MarginalQ = theano.printing.Print("MarginalQ")(MarginalQ)
    #TotalP = theano.printing.Print("TotalP")(TotalP)
    #TotalQ = theano.printing.Print("TotalQ")(TotalQ)
    
    return JoinPQ(MarginalP, TotalP, MarginalQ, TotalQ, C)
    


def PrepareOracleForBCNM(domain, Samples, Prior, Mean, C, Alpha, Lambda, Support, Mask):
    Sum = PrepareCountingOracle(domain, Samples, Prior, Lambda, C, Support, Mask)
    LogSum = T.log(Sum)
    DistanceToMean = LogSum - T.dot(Lambda, Mean)
    Regularizer = Alpha * T.dot(Lambda, Lambda)
    Value = DistanceToMean + Regularizer
    Gradient = T.grad(Value, Lambda)
    Hessian = T.hessian(Value, Lambda)
    return Value, Gradient, Hessian

class OracleForBCNM(object):
    def __init__(self, domain):
        nX = domain.numberOfFeatures
        sizes = domain.getSizes()
        nY = max(sizes)
        nZ = domain.dimension
    
        support = np.zeros((nX, nY, nZ))
        mask = np.zeros((nX, nY))
    
        start = 0
        for i, uniqueValues in enumerate(domain):
            n, m = uniqueValues.shape
            support[i, :n, start:start+m] = uniqueValues
            start += m
            mask[i, :n] = 1
        Support = shared(support)
        Mask = shared(mask)
        
        dualDimension = start
        maxFeature = nY
        
        self.dualDimension = dualDimension
        self.maxFeature = maxFeature
        self.numFeature = nX
        numFeature = self.numFeature
        
        Samples = shared(np.random.randn(1, dualDimension))
        Prior = shared(np.zeros(1))
        Mean = shared(np.random.randn(dualDimension))
        C = shared(0.0)
        Alpha = shared(0.0)
        
        
        Lambda = T.vector("lambda")
    
        Output = PrepareOracleForBCNM(domain, Samples, Prior, Mean, C, Alpha, Lambda, Support, Mask)

        
        self.Samples = Samples
        self.Prior = Prior
        self.Mean = Mean
        self.C = C
        self.Alpha = Alpha
        self.Support = Support
        self.Mask = Mask
        
        self.Lambda = Lambda
        self.function = theano.function([Lambda], Output, allow_input_downcast=True)#, profile=True)
                        
    def __call__(self, Lambda):
        return self.function(Lambda)
   
    def set(self, Samples=None, Prior=None, C=None, Alpha=None, Mean=None):
        if Samples is not None:
            self.Samples.set_value(Samples)
            
        if Prior is not None:
            self.Prior.set_value(Prior)
        
        
        if C is not None:
            self.C.set_value(C)
        
        if Alpha is not None:
            self.Alpha.set_value(Alpha)
        
        if Mean is not None:
            self.Mean.set_value(Mean)
            

def prepareOracleForBCNM(domain):
    #, weights=None, samples=None, prior=None, mean=None, c=None, alpha=None):
    #h = hash(featureDescriptions, isUniform, weights, samples, prior, mean, c, alpha)
    h = hash(domain)
    name = os.path.join(modelFolder, "oracleForBCNM-{hash}.model".format(hash=h))
    if os.path.exists(name):
        with open(name, 'rb') as file:
            f = pickle.load(file)
        return f
    f = OracleForBCNM(domain)
    with open(name, 'wb') as file:
        pickle.dump(f, file, protocol=pickle.HIGHEST_PROTOCOL)
    return f


def PrepareMarginalProbabilities(domain, Prior, C, Lambda, Samples, Support, Mask):
    TotalP = GetMarginalP0(Prior)
    TotalQ = GetMarginalQ0(domain)
    Qijs = GetQij(domain, Lambda, Support, Mask)
    LogQMarginals = GetLogMarginalsQ(Qijs)
    LogQijs = T.log(Qijs)    
    
    TotalPriorMass = PrepareMarginal(Samples, Prior, Lambda) 
    TotalQMass = T.exp(T.sum(LogQMarginals)) 
    TotalMass = JoinPQ(TotalPriorMass, TotalP, TotalQMass, TotalQ, C)
    
    return TotalP, TotalQ, TotalMass, LogQMarginals, LogQijs
    
    

class MarginalProbabilities(object):
    def __init__(self, domain):
        nX = domain.numberOfFeatures
        sizes = domain.getSizes()
        nY = max(sizes)
        nZ = domain.dimension
    
        support = np.zeros((nX, nY, nZ))
        mask = np.zeros((nX, nY))
    
        start = 0
        for i, uniqueValues in enumerate(domain):
            n, m = uniqueValues.shape
            support[i, :n, start:start+m] = uniqueValues
            start += m
            mask[i, :n] = 1
        Support = self.Support = T.tensor3()
        Mask = self.Mask = T.matrix()
        
        self.support = support
        self.mask = mask
        

        dualDimension = nZ
        maxFeature = nY
        
        self.dualDimension = dualDimension
        self.maxFeature = maxFeature
        self.numFeature = nX
        numFeature = self.numFeature
        
#        Samples = shared(np.random.randn(1, dualDimension))
        self.samples = np.random.randn(1, dualDimension)
#        Prior = shared(np.zeros(1))
        self.prior = np.zeros(1)
#        Lambda = shared(np.random.randn(dualDimension))
        self.lambda_ = np.random.randn(dualDimension)
#        C = shared(0.0)
        Samples = T.matrix()
        Prior = T.vector()
        Lambda = T.vector()
        C = T.scalar()
        
#        Weights = shared(np.zeros((numFeature, maxFeature)))

                         
        
        Output = PrepareMarginalProbabilities(domain, Prior, C, Lambda, Samples, Support, Mask)
        
        self.Samples = Samples
        self.Prior = Prior
        self.Lambda = Lambda
        self.C = C
        self.c = 0
        self.Support = Support
        self.Mask = Mask
        
        self.function = theano.function([Samples, Prior, Lambda, C, Support, Mask], Output, allow_input_downcast=True)
                        
    def __call__(self, priorMass, coordinates):
        totalP = self.totalP
        totalQ = self.totalQ
        totalMass = self.totalMass
        logQMarginals = self.logQMarginals
        logQijs = self.logQijs
        numFeature = self.numFeature
        C = self.c
        
        logQMasses = np.zeros_like(logQMarginals)
        logQMasses[:] = logQijs[np.arange(numFeature), coordinates]
        
        logQMasses[coordinates<0] = logQMarginals[coordinates<0]
        qMass = np.exp(np.sum(logQMasses))
        
        marginalMass = JoinPQ(priorMass, totalP, qMass, totalQ, C)
        
        return marginalMass / totalMass
                              
   
    def set(self, Samples=None, Prior=None, Lambda=None, C=None):
        if Samples is not None:
            self.samples = Samples
#            self.Samples.set_value(Samples)
            
        if Prior is not None:
            self.prior = Prior
#            self.Prior.set_value(Prior)
        
        if Lambda is not None:
            self.lambda_ = Lambda
#            self.Lambda.set_value(Lambda)
        
        if C is not None:
#            self.C.set_value(C)
            self.c = C
        
        totalP, totalQ, totalMass, logQMarginals, logQijs = self.function(self.samples, self.prior, self.lambda_, self.c,  self.support, self.mask)
        self.totalP = totalP
        self.totalQ = totalQ
        self.totalMass = totalMass
        self.logQMarginals = logQMarginals
        self.logQijs = logQijs
            

def prepareMarginalProbabilities(domain):
    #, weights=None, lambda_=None, c=None, prior=None, samples=None):
    #h = hash(featureDescriptions, isUniform, weights, lambda_, c, prior, samples)
    h = hash(domain)
    name = os.path.join(modelFolder, "marginalProbabilities-{hash}.model".format(hash=h))
    if os.path.exists(name):
        with open(name, 'rb') as file:
            f = pickle.load(file)
        return f
    f = MarginalProbabilities(domain)
    with open(name, 'wb') as file:
        pickle.dump(f, file, protocol=pickle.HIGHEST_PROTOCOL)
    return f

    
    
