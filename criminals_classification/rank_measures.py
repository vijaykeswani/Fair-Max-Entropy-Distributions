import math
import random
import numpy as np

def avg_swaps(rank1,rank2):
    n=len(rank1)
    sum=0
    max_sum=0
    for i in range(n):
        for j in range(n):
            if rank1[i]<rank1[j] and rank2[i]>rank2[j]: sum+=rank1[j]-rank1[i]
            if rank1[i]<rank1[j]: max_sum+=rank1[j]-rank1[i]
    return sum*1.0/max_sum
    
def expected_avg_swaps(rank1,rank2):
    rs=random.getstate()
    #assure uniform tiebreaking
    random.seed(1)
    iter=50
    vals=[]
    for n_iter in range(iter):
        rank2p=[(e,random.randint(0,10**10)) for e in rank2]
        vals.append(avg_swaps(rank1,rank2p))
    vals=np.array(vals)
    random.setstate(rs)
    return (np.mean(vals),np.var(vals))

def KLdiv(p,q):
    n=len(p)
    result=0.0
    for i in range(n):
        x=p[i]
        y=q[i]
        if x==0.0: x=1e-9
        if y==0.0: y=1e-9
        result+=x*math.log(x/y)
    return result

def rND(ranking):
# ranking is a 0-1 list, ranking[i] indicates whether the ith element in the ranking is in the protected group or not
# ranking needs to have at least 10 elements
    Z=0
    total_protected=0.0
    for i in ranking:
        total_protected+=i
    n=len(ranking)
    protected_so_far=0.0
    result=0.0
    for i in range(n):
        protected_so_far+=ranking[i]
        result+=1/math.log(i+2)*abs(protected_so_far/(i+1) - total_protected/n)
        Z+=1/math.log(i+2)*max(total_protected/n,(1.0-total_protected/n))
    return result/Z
    
def expected_rND(rank_scores,protected):
    rs=random.getstate()
    #assure uniform tiebreaking
    random.seed(1)
    iter=50
    vals=[]
    for n_iter in range(iter):
        rank_scores_rand=[(e,random.randint(0,10**10)) for e in rank_scores]
        rank_prot=zip(rank_scores_rand,protected)
        rank_prot.sort()
        to_score=[y for (x,y) in rank_prot]
        vals.append(rND(to_score))
    vals=np.array(vals)
    random.setstate(rs)
    return (np.mean(vals),np.var(vals))
        
def rKL(ranking):
# ranking is a 0-1 list, ranking[i] indicates whether the ith element in the ranking is in the protected group or not
# ranking needs to have at least 10 elements   
    Z=0
    total_protected=0.0
    for i in ranking:
        total_protected+=i
    n=len(ranking)
    protected_so_far=0.0
    result=0.0
    for i in range(n):
        protected_so_far+=ranking[i]
        if i%10==9:
            p=[protected_so_far/i,1-protected_so_far/i]
            q=[total_protected/n,1-total_protected/n]
            result+=1/math.log(i)*KLdiv(p,q)
    return result
    


