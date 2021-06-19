#一个.代表从当前目录导入
from utils.utils import *
import time
from data import load_cifar10
import math


def register_master(grpc_server):
    a=Master()
    tf_pb2_grpc.add_masterServicer_to_server(a, grpc_server)
    return a

class Master(tf_pb2_grpc.masterServicer):
    def __init__(self):

        self.ps = None #用来存储对ps连接通道
        self.workers={}#用来存储对worker的连接通道
        self.time_next={}#用来记录下次迭代所需要的时间,键为ip地址，值为下一次的迭代时间
        self.finish_ip=[] #用来记录已经完成的任务的ip,这里的ip是动态变化
        self.tmp_ip=[]
        self.is_worker_finish=False #用来判断是否有worker完成工作的信号
        self.is_worker_aggrate=False#用来判断是否要进行梯度聚合
        self.is_ps_aggrate_finished=True #用来做判断是否要再次通知worker训练
        self.is_aggrating_ps=False #用来记录是否正在聚合任务中，这个参数主要解决在参数聚合中有worker完成了任务


        self.count=0 #用来记录有多少worker通知自己完成任务
        self.worker_num=18 #用来记录参数计算的worker的数量，方便调用


    def register_cluster(self, request, context):
        ps_ipadrr=request.ps
        self.ps=connect_ps(ps_ipadrr)
        print(self.ps.register_cluster(request).mes)
        for ws_addr in request.workers:
            key_ip = ws_addr.split(":")[0]
            # 以键值对的方式存储
            self.workers[key_ip] = connect_worker(ws_addr)
            print(self.workers[key_ip].register_cluster(request).mes)
        #给worker的数量进行初始化
        #self.worker_num = len(self.workers.keys())
        #先调用send函数把数据发送过去
        self.send()
        return tf_pb2.status(code=200, mes="master register cluster success")

    def send(self):
         (x_train, y_train),(x_test, y_test)=load_cifar10.load_cifar("../opt_syn/data/cifar-10-batches-py")

         #为了方便数据的在不同的机器之间进行传输
         num_workers = len(self.workers.keys())

         #计算每个节点获得多少条数据,我们把多余的数据给到最后一个worker
         data_size = int(x_train.shape[0]/num_workers)

         #先给n个节点发送数据，因为最后一个节点的数据会多些
         j=0
         for  i in self.workers.keys():
            response = self.workers[i].send_train_data.future(tf_pb2.data(index=j,
              index_gap=data_size)).result()
            j=j+1
            print(response.mes)

   #接收Client来的通知开始训练
    def notice_train(self, request, context):
        print(request.notice)
        for ip in self.workers:
             self.workers[ip].notice_start_train(tf_pb2.note(notice="master通知开始训练"))
        return tf_pb2.status(code=200,mes="收到训练的任务，并完成")


    def worker_notice_master_finished(self, request, context):
        while self.is_aggrating_ps==False:
           time.sleep(0.3)

        self.finish_ip.append(request.w_ip)

        self.count = self.count + 1

        if (self.count%math.floor(self.worker_num*0.8))==0:
           self.is_aggrating_ps=False
           self.is_worker_finish = True

        return tf_pb2.status(code=200,mes="ps收到了worker完成训练的通知")

    def master_notice_ps(self):
                self.tmp_ip = self.finish_ip
                self.finish_ip = []
                # 这个向ps发送聚合的通知，并把要聚合的ip过去
                res = self.ps.master_notice_ps_to_pull_data_from_worker(tf_pb2.worker_fin_ip(w_ip=self.tmp_ip))
                print(res.mes)



    def master_notice_ps_aggrate(self):
         self.ps.master_notice_ps_aggregate(tf_pb2.note(notice="master通知聚合参数"))


    def notice_worker_again_train(self):

             for ip in self.tmp_ip:
               self.workers[ip].notice_start_train(tf_pb2.note(notice="master通知再次开始训练"))
             self.is_aggrating_ps=True

    #供ps通知master完成了梯度聚合的任务
    def ps_notice_master_pull_data_finished(self, request, context):
        print(request.notice)
        self.is_ps_aggrate_finished=True
        return tf_pb2.status(code=200,mes="master收到了ps完成数据pull")






