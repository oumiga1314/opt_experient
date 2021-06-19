from scipy.cluster.vq import *
#采用方差边界值来计算
def squre(data,data_temp,k):
    loss = 0.0
    if data_temp==None:
        return
    temp=sorted(data_temp)
    print(temp)
    sum=0.0
    m=0
    for i in data:
        if i<=temp[0]:
            sum=sum+temp[0]-i
            m=m+1
    res = abs(temp[0]-temp[1])
    sum=sum/m
    #res代表边界差
    #sum代表平均等待时间
    #m代表该类中有多少个worker
    return  temp[0],res,sum,m




def cluster_loss(num,a):
    cluster_num = num
    #res代表聚类的中心选取
    #idx代表聚类后的结果，
    res, idx = kmeans2(a, cluster_num, iter=200, minit="points")
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
    return squre(a,x_find,num)

def select_wait(xx,k,agg_time,threshold):
    """
    :param x:带表传入的时间
    :param k: 代表传入的候选k
    :param agg_time: 代表平均聚合一个梯度的时间
    :return:
    """
    a=sorted(xx)
    print("select"+str(a))
    tmp=10000
    tmp_k=1
    #如果k=0返回里面最大的完成时间，这个可以避免如果有的节点死掉，造成整个系统的暂停
    if k>=threshold:
        candicate_k=[k-3,k-2,k-3]
    elif k<=3:
        candicate_k=[2,3,4]
    else:
         candicate_k=[k-1,k,k+1]
    for ii in candicate_k:
        wait_opt,bian,av,m = cluster_loss(ii,a)
        print("bian"+str(bian))
        print("av:"+str(av))
        if av < tmp and bian>m*agg_time:
            tmp = wait_opt
            tmp_k=ii
    if a[-1]<=len(a)*agg_time*0.3:
        tmp=10
        tmp_k=1
    print("res")
    print(tmp)
    print(tmp_k)
    print("res")
    return tmp,tmp_k
# data =[3,4,5,3.4,4.5,4.5,6,6.5,9,10,11,10.5,18,19,20,18.5,17,17.5]
#
# select_wait(data,3,0.3,7)