import numpy as np
from datetime import datetime
import criminals_classification.read_dataset
import criminals_classification.svm_rank
import random
import criminals_classification.rank_measures
import numpy.linalg as la


    
def normalize_sd(nrecords):
    n=len(nrecords)
    d=len(nrecords[0])
    for j in range(d):
        sum=0.0
        for i in range(n):
            sum+=nrecords[i][j]
        mean=sum/n
        var=0.0
        for i in range(n):
            nrecords[i][j]-=mean
            var+=nrecords[i][j]**2
        std_d=np.sqrt(var/n)
        for i in range(n):
            nrecords[i][j]/=std_d
            
def normalize(nrecords):
    n=len(nrecords)
    d=len(nrecords[0])
    for j in range(d):
        mx=0.0
        for i in range(n):
            mx=max(mx,abs(nrecords[i][j]))
        for i in range(n):
            nrecords[i][j]/=mx
    
def regularize_data(X,delta):
    n=X.shape[0]
    K=np.dot(X,X.T)
    Kp=K+delta*np.eye(n)
    #(u,s,v)=la.svd(Kp)
    #print(s)
    L=la.cholesky(K+delta*np.eye(n))
    return L

def text_fields_to_features(tf):
    (lower,upper)=(3,500)
    bag=[]
    for s in tf:
        g="".join([ c if c.isalpha() else " " for c in s ])
        g=g.lower()
        l=map(str.strip,g.split())
        l=filter(lambda x: len(x)>1,l)
        bag.extend(l)
    take=[]
    ubag=list(set(bag))
    for e in ubag:
        ec=bag.count(e)
        if lower<=ec and ec<=upper:
            take.append(e)
    take.sort()
    features=[]
    for s in tf:
        g="".join([ c if c.isalpha() else " " for c in s ])
        g=g.lower()
        l=map(str.strip,g.split())
        v=[]
        for i in range(len(take)):
            if take[i] in l: v.append(1.0)
            else: v.append(0.0)
        features.append(v)
    print('text features: %d'%len(take))     
    return features

def data_to_vectors(data,types,head,columns_to_keep):
    d=len(data[0])
    ndata=[[] for e in data]
    for c in range(len(data[0])):
        if head[c] not in columns_to_keep: continue
        if types[c]=='text':
            text_fields=[r[c] for r in data]
            text_features=text_fields_to_features(text_fields)
            for i in range(len(data)):
                ndata[i].extend(text_features[i])
        elif types[c]=='num':
            min_val=0        
            for i in range(len(data)):
                min_val=min(min_val,float(data[i][c]))
            for i in range(len(data)):
                x=data[i][c]
                if min_val==-1: 
                    x+=1
                ndata[i].append(float(x))
        else:
            sval=set()
            for i in range(len(data)):
                sval.add(data[i][c])
            sval=list(sval)
            vals=len(sval)
            for i in range(len(data)):
                indicator=[0.0]*vals
                for k in range(vals):
                    if sval[k]==data[i][c]:
                        indicator[k]=1.0
                ndata[i]=ndata[i]+indicator
    normalize(ndata)
    print(len(ndata))
    print(len(ndata[0]))
    return ndata
    

