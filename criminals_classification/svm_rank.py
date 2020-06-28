# -*- coding: utf-8 -*-
import numpy as np
import os

#set to a lower value for SVM to run faster
ACCURACY_SVM=30.0

def save_to_file(data,ranks,path):
    f=open(path,'w')
    n=len(data)
    d=len(data[0])
    for i in range(n):
        s=str(ranks[i])+' qid:1 '
        for j in range(d):
            s=s+str(j+1)+':'+str(data[i][j])+' '
        f.write(s+'\n')
    f.close()
    

def generate_train_svm(data,ranks):
    save_to_file(data,ranks,'tmp\\train.dat')
    n=len(data)
    acc=ACCURACY_SVM/n
    os.system('svm_rank_learn -c %.4f  -g 1 -v 0 tmp\\train.dat tmp\\data_model.dat'%acc)
    
def generate_test_svm(data,ranks):
    save_to_file(data,ranks,'tmp\\test.dat')
    os.system('svm_rank_classify  -v 0 tmp\\test.dat tmp\\data_model.dat tmp\\predictions.dat')

def read_ranking(path):
    f=open(path,'r')
    g=f.readlines()
    f.close()
    g=map(float,g)
    return g


def generate_ranking(train,train_ranks,test,test_ranks):
    generate_train_svm(train,train_ranks)
    generate_test_svm(test,test_ranks)
    return read_ranking('tmp\\predictions.dat')
