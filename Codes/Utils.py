import hashlib
import aif360
import sys
import numpy as np
sys.path.insert(0, "../")
from FairMaxEnt.domain import Domain
from FairMaxEnt.memory import MemoryTrie

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_compas
from aif360.datasets import CompasDataset

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.datasets import AdultDataset
import sklearn
import itertools
import scipy
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from criminals_classification.read_dataset import read_dataset
from criminals_classification.prepare import data_to_vectors


def hash(*args):
    hashSHA512 = hashlib.sha512()
    hashSHA512.update(str(args).encode("utf-8"))
    return hashSHA512.hexdigest()


def getSmallCompasDomain():
    domainArray = [[0, 1], # sex
                    [0, 1], # race
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], # age
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]], # priors count
                    [[1, 0], [0, 1]], # c charge degree
                    [0, 1] # label
                  ]
    return domainArray

def getSmallCompasDataset():

    dataset = CompasDataset()
    dataset_orig = load_preproc_data_compas(['sex'])
    
    features = ['sex', 'race', 'age', 'priors_count', 'c_charge_degree']
    domainArray = getSmallCompasDomain()
    features.append(dataset_orig.label_names[0])

    simpleDomain = Domain(features, domainArray)
    labels = [y[0] for y in dataset_orig.labels]
    
    simpleSamples = dataset_orig.features
    simpleSamples = np.c_[simpleSamples, labels]
    
    return simpleDomain, simpleSamples



def getLargeCompasDataset():
    dataset = CompasDataset()
    dataset_orig = load_preproc_data_compas(['sex'])

    (head,types,records)=read_dataset()   
    records = np.array(records)


    head,types = head[:-1], types[:-1]
    records = np.delete(records, -1, axis=1)


    def reorder(record):
        record.pop(11)
        record.append(record.pop(10))
        return record

    records = [reorder(list(record)) for record in records]
    head = reorder(head)

    records = np.array(records)

    def getHotEncoding(index, n):
        temp = [0]*n
        temp[index] = 1
        return tuple(temp)

    races = list(set(records[:, 2]))
    print (races)
    fmo = list(set(records[:, 9]))
    nrecords = []
    domainArray = [set([]) for h in head]
    for record in records:
        temp = []
        for j, (r, h) in enumerate(zip(record, head)):
            if h == "sex":
                if r == 'Male':
                    entry = 1
                else:
                    entry = 0
            elif h == "age":
                # age
                age = int(r)
                if age <= 25:
                    entry = 0
                elif age <=65:
                    entry = 1
                else:
                    entry = 2
            elif h == "race":
                # race
                if races.index(r) == 3:
                    entry = 1
                else: 
                    entry = 0
                #entry = getHotEncoding(races.index(r), len(races))
            elif h == "priors_count":
                # priors count
                priors = int(r)
                if priors <= 0:
                    entry = 0
                elif priors <=10:
                    entry = 1
                elif priors <=20:
                    entry = 2
                elif priors <=30:
                    entry = 3
                elif priors <=40:
                    entry = 4
                else:
                    entry = 5
            elif h == "days_in_jail":
                # months in jail    
                months = int(r)/12.0
                if months <= 0:
                    entry = 0
                elif months <=3:
                    entry = 1
                elif months <=6:
                    entry = 2
                elif months <=12:
                    entry = 3
                elif months <=24:
                    entry = 4
                elif months <=48:
                    entry = 5
                elif months <=60:
                    entry = 6
                else:
                    entry = 7
            elif h == "c_charge_degree":
                entry = fmo.index(r)
            else:
                entry = float(r)
            domainArray[j].add(entry)
            try:
                temp.extend(entry)
            except:
                temp.append(entry)
        nrecords.append(np.array(temp))

    nrecords = np.array(nrecords)

    for index in range(19):
        temp = []
        for record in nrecords:
            temp.append(record[index])

    domainArray = [np.array(list(uvs)) for uvs in domainArray]


    simpleDomain = Domain(head, domainArray)

    return simpleDomain, nrecords
    

def getAdultDomain():
    domainArray = [[0, 1], # race
                    [0, 1], # sex
                    [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0], 
                    [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]], # age
                    [[1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], 
                     [0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0], 
                     [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1]], # education years
                    [0, 1], # label
                  ]
    return domainArray

def getAdultDataset():

    dataset = AdultDataset()
    dataset_orig = load_preproc_data_adult(['sex'])

    features = ['race', 'sex', 'age decade', 'education years']
    domainArray = getAdultDomain()
    
    features.append(dataset_orig.label_names[0])
    simpleDomain = Domain(features, domainArray)

    labels = [y[0] for y in dataset_orig.labels]

    simpleSamples = dataset_orig.features
    simpleSamples = np.c_[simpleSamples, labels]    
    return simpleDomain, simpleSamples

    
def getDisparateImpact(dataset, sens_attr = 0):
    labelIndex = len(dataset[0])-1
    
    y1z1 = sum(dataset[:,-1] * dataset[:,sens_attr])
    y1z0 = sum(dataset[:,-1] * (1-dataset[:,sens_attr]))
    z1 = sum(dataset[:,sens_attr])
    z0 = len(dataset) - z1

    return min((y1z0/z0)/(y1z1/z1), (y1z1/z1)/(y1z0/z0)) 

def getGenderRatio(dataset, sens_attr = 0):
    z1 = sum(dataset[:,sens_attr])
    z0 = len(dataset) - z1
    return min(z0/z1, z1/z0) 


def getTrainAndTestData(dataset, i):
    indices = list(range(len(dataset)))
    limit = int(len(dataset)/5.0)
    fold = [[],[],[],[],[]]
    fold[0] = indices[0:limit]
    fold[1] = indices[limit:2*limit]
    fold[2] = indices[2*limit:3*limit]
    fold[3] = indices[3*limit:4*limit]
    fold[4] = indices[4*limit:]
    
    testData = []
    for ind in fold[i]:
        testData.append(dataset[ind])
    trainData = []
    for j in range(5):
        if j!=i:
            for ind in fold[j]:
                trainData.append(dataset[ind])
            
    return np.array(trainData), np.array(testData)



def getDistribution(dataset, domain):
    y1z1, y1z0, z1, z0 = 0, 0, 0, 0
    p = [0]*len(domain)

    for elem in dataset:
        ind = -1
        for i, elem2 in enumerate(domain):
            if list(elem) == list(elem2):
                ind = i
                break
            
        p[ind] += 1
    p = np.array(p)
    return p/len(dataset)

def getDomain(domainArray):
    domain1 = list(itertools.product(*domainArray))
    domain = []

    for elem in domain1:
        elem2 = []
        for j in elem:
            if isinstance(j, (list,)):
                for k in j:
                    elem2.append(k)
            else:
                elem2.append(j)
        domain.append(elem2)
    return domain
    

def getUtility(dataset, rawDataDist, domain):
    dist = getDistribution(dataset, domain)
    return scipy.stats.entropy(dist, rawDataDist)


def getDisparateImpactClf(dataset, yL, sens_attr=0):
    y1z1, y1z0, z1, z0 = 0, 0, 0, 0
    y1z1 = sum(yL * dataset[:,sens_attr])
    y1z0 = sum(yL * (1-dataset[:,sens_attr]))
    z1 = sum(dataset[:,sens_attr])
    z0 = len(dataset) - z1

    if y1z0 == 0:
        return (y1z0/z0)/(y1z1/z1)
    if y1z1 == 0:
        return (y1z1/z1)/(y1z0/z0)
    
    return min((y1z0/z0)/(y1z1/z1), (y1z1/z1)/(y1z0/z0)) 

def getClfAccAndDI(dataset, testData, sens_attr, clf = GaussianNB()):
    
    labelIndex = len(dataset[0])-1
    X, Y = [], []
    for elem in dataset:
        X.append(elem[:labelIndex])
        Y.append(elem[labelIndex])
        
    clf.fit(X,Y)
    
    X, Y = [], []
    for elem in testData:
        X.append(elem[:labelIndex])
        Y.append(elem[labelIndex])
        
    Y_pred = clf.predict(X)
    
    accuracy = (Y_pred == Y).sum()/len(Y)
    
    Y_pred = clf.predict_proba(X)[:,1]
    DI = getDisparateImpactClf(testData, Y_pred, sens_attr)
    
    return accuracy, DI


def getCorr(dataset):
    n = len(dataset[0])
    corr = {}
    for i in range(n):
        c1 = dataset[:,i]
        for j in range(n):
            c2 = dataset[:,j]
            corr[(i,j)] = scipy.stats.pearsonr(c1, c2)[0]
    return corr

def getCorrUtility(dataset, rawCorr):
    corr = getCorr(dataset)
    total = 0
    for key in corr.keys():
        total += (corr[key] - rawCorr[key])**2
    return total

def cPlot(values, ylabel, title=""):
    ys = []
    for key in range(6):
        a = np.array(values[key]).reshape(11, 4)
        y = np.array([(np.mean(row), np.std(row)) for row in a])
        ys.append(y)

    plt.ylabel(ylabel, fontsize=13)
    plt.xlabel("C", fontsize=16)
    plt.title(title)
    
    plt.errorbar(np.arange(0.0, 1.1, 0.1), ys[0][:,0], ys[0][:,1], fmt="o-", color='blue', label="Prior $q_C^{d}$, Expected Value $\\theta^{d}$")
    plt.errorbar(np.arange(0.0, 1.1, 0.1), ys[1][:,0], ys[1][:,1], fmt="x-", color='blue', label="Prior $q_C^{w}$, Expected Value $\\theta^{d}$")
    plt.errorbar(np.arange(0.0, 1.1, 0.1), ys[2][:,0], ys[2][:,1], fmt="o-", color='red', label="Prior $q_C^{d}$, Expected Value $\\theta^{b}$")
    plt.errorbar(np.arange(0.0, 1.1, 0.1), ys[3][:,0], ys[3][:,1], fmt="x-", color='red', label="Prior $q_C^{w}$, Expected Value $\\theta^{b}$")
    plt.errorbar(np.arange(0.0, 1.1, 0.1), ys[4][:,0], ys[4][:,1], fmt="o-", color='green', label="Prior $q_C^{d}$, Expected Value $\\theta^{w}$")
    plt.errorbar(np.arange(0.0, 1.1, 0.1), ys[5][:,0], ys[5][:,1], fmt="x-", color='green', label="Prior $q_C^{w}$, Expected Value $\\theta^{w}$")

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
              ncol=3, fancybox=True, shadow=True, fontsize=12)

    plt.show()        