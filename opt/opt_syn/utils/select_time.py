import numpy as np

from sklearn.cluster import KMeans

# a =[]
# a=sorted(a)
# print(a)



from scipy.cluster.vq import *

import matplotlib.pyplot as plt

#采用方差边界值来计算
def squre(data_):
    loss = 0.0
    if data_==None:
        return loss
    data=sorted(data_[1:-1])
    for i,j in zip(data[:-1],data[1:]):
        loss=loss+abs(i-j)
    return loss/(len(data)/2)

#采用等待时间法来进行计算
def wait_time(data_):
    loss = 0
    if(data_==None):
        return loss
    data=sorted(data_)
    last=data[-1]
    for i in data[0:-1]:
        loss=loss+(last-i)
    return loss

def cluster_loss(num,a):
    cluster_num = num
    res, idx = kmeans2(a, cluster_num, iter=200, minit="points")
    x = []
    x_find=[]
    for i in range(cluster_num):
        x = []
        #k为了计算有多少台机器
        k = 0
        for j in idx:
            if i == j:
                x.append(a[k])
            k = k + 1
        #为了找出下一次的最佳同步点
        if x:
          x_find.append(x[-1])
    min_find=min(x_find)
    return min_find
tmp=0
candicate_k=[]
def select_wait(xx,k):
    """
    :param x:带表传入的时间
    :param k: 代表传入的候选k
    :return:
    """
    a=sorted(xx)
    print("select"+str(a))
    tmp=0
    candicate_k=[]
    #如果k=0返回里面最大的完成时间，这个可以避免如果有的节点死掉，造成整个系统的暂停
    if k==1:
        return a[-1]
    else:
        if k<=3:
            candicate_k=[2,3,4]
        else:
            candicate_k=[k-1,k,k+1]
    for ii in candicate_k:

        x = cluster_loss(ii,a)
        if tmp < x:
            tmp = x
    return tmp
