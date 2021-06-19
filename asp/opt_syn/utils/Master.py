#一个.代表从当前目录导入
from opt_syn.utils.utils import *
import time
from data import load_cifar10
import threading
import math
import random
from utils.select_time import select_wait
from utils.select_candiate_key import candicate_key

def register_master(grpc_server):
    a=Master1()
    tf_pb2_grpc.add_masterServicer_to_server(a, grpc_server)
    return a

class Master1(tf_pb2_grpc.masterServicer):
    def __init__(self):

        self.ps = None #用来存储对ps连接通道
        self.workers={}#用来存储对worker的连接通道
        self.time_next={}#用来记录下次迭代所需要的时间,键为ip地址，值为下一次的迭代时间
        self.finish_ip=[] #用来记录已经完成的任务的ip,这里的ip是动态变化
        self.tmp_ip=[]
        self.is_worker_finish=False #用来判断是否有worker完成工作的信号
        self.is_worker_aggrate=False#用来判断是否要进行梯度聚合
        self.is_ps_aggrate_finished=False #用来做判断是否要再次通知worker训练
        self.is_opt_time=False  #用来开启选择计算等待的时间
        self.is_aggrating_ps=False #用来记录是否正在聚合任务中，这个参数主要解决在参数聚合中有worker完成了任务
        self.time_next_next={} #用来记录下一次的下一次到达的时间，这个也是为了处理在参数聚合中有worker完成了任务

        self.k = 1 # 用来调整候选k
        self.flag = True  # 用来调整k是上行还是下行，
        self.threshold = 0,  # 用来计算候选k的上限
        self.count=0 #用来记录有多少worker通知自己完成任务
        self.count_print=0#仅仅用来进行测试
        self.wait_syn_time=0#定义用来接收要等待多长时间来进行下一次迭代
        self.record_syn_start=0#用来记录同步要开始的时间

        #要进行完全同步的一些参数（就是所有worker进行一次同步）
        self.syn_count=0 #用来统计此次需要进行所有worker同步的数量
        self.worker_num=0#用来记录参数计算的worker的数量，方便调用

        #为了对不参与本次同步的worker的完成时间进行更新
        self.all_worker_ip=[]

        self.master_is_free=True# 表示master正在空闲中，

        self.master_opt_syn=False #为了防止由于等待时间错误，而所有worker均已经到达，没有办法再通知继续工作，而设置此变量

    def register_cluster(self, request, context):
        ps_ipadrr=request.ps
        self.ps=connect_ps(ps_ipadrr)
        print(self.ps.register_cluster(request).mes)
        self.time_next={}



        for ws_addr in request.workers:
            key_ip = ws_addr.split(":")[0]
            # 以键值对的方式存储
            self.workers[key_ip] = connect_worker(ws_addr)
            #对到达的下下一次时间进行初始化
            self.time_next_next[key_ip]=0
            print(self.workers[key_ip].register_cluster(request).mes)

        #给worker的数量进行初始化
        self.worker_num = len(self.workers.keys())
        # 用来初始化k的上限
        # self.threshold=math.ceil(len(self.workers.keys())*0.5)
        self.threshold = math.floor(18*0.35)
        self.all_worker_ip=list(self.workers.keys())

        #先调用send函数把数据发送过去
        self.send()
        return tf_pb2.status(code=200, mes="master register cluster success")

    def send(self):
         (x_train, y_train),(x_test, y_test)=load_cifar10.load_cifar("../opt_syn/data/cifar-10-batches-py")
         #为了方便数据的在不同的机器之间进行传输
         num_workers = len(self.workers.keys())

         #计算每个节点获得多少条数据,我们把多余的数据给到最后一个worker
         data_size = int(x_train.shape[0]/num_workers)
         #把数据有多少赋给dim
         dim=x_train.shape[1]
         #先给n个节点发送数据，因为最后一个节点的数据会多些
         j=0
         for  i in self.workers.keys():
            response = self.workers[i].send_train_data(tf_pb2.data(index=j,
              index_gap=data_size))
            j=j+1
            print(response.mes)

   #接收Client来的通知开始训练
    def notice_train(self, request, context):
        print(request.notice)
        for ip in self.workers:
             self.workers[ip].notice_start_train(tf_pb2.note(notice="master通知开始训练"))
        return tf_pb2.status(code=200,mes="收到训练的任务，并完成")


    def worker_notice_master_finished(self, request, context):
        #在master还没有收到ps聚合梯度完成的时候，不允许其他的worker进入，一直在等待状态
        while self.master_is_free==False:
            time.sleep(0.05)
        if self.k == 1:
               self.count = self.count + 1
                #因为这个worker提前到到达，所以要把下一次迭代时间置为0
               self.count_print = self.count_print + 1
               print(self.count_print)
               self.time_next[request.w_ip] = request.next_time
               self.finish_ip.append(request.w_ip)
               #还需要记录下下一次到达的worker的时间,暂且还没有用到
               self.time_next_next[request.w_ip] = request.next_time
               self.is_worker_finish = True

        else:
            self.count_print = self.count_print + 1
            print(self.count_print)
            self.finish_ip.append(request.w_ip)
            self.time_next[request.w_ip]=request.next_time

            #self.is_worker_finish = True





        return tf_pb2.status(code=200,mes="ps收到了worker完成训练的通知")

    def master_notice_ps(self):

        #在这个地方添加条件
        #如果k==1说明此时需要进行完全同步，要等到所有worker到达
        if self.k==1:
            if self.count==self.worker_num:
                self.master_is_free=False
                self.count=0
                self.tmp_ip=self.finish_ip
                self.finish_ip=[]
                #向ps通知，通知ps聚合梯度，并且把要聚合的worker的ip带过去
                res=self.ps.master_notice_ps_to_pull_data_from_worker(tf_pb2.worker_fin_ip(w_ip=self.tmp_ip))
                print(res.mes)

        else:
            pass

    def master_notice_ps_aggrate(self):
        lock2=threading.Lock()
        lock2.acquire(True)
        self.ps. master_notice_ps_aggregate(tf_pb2.note(notice="master通知聚合参数"))
        lock2.release()

    def notice_worker_again_train(self):
             for ip in self.tmp_ip:
               self.workers[ip].notice_start_train(tf_pb2.note(notice="master通知再次开始训练"))

             # #接下来调整候选k
             self.k,self.flag=candicate_key().candiate(self.k,self.threshold,self.flag)
             print("k="+str(self.k))
             print("flag"+str(self.flag))

             #接下来计算最佳同步点，是在哪个时刻
             #1.判断候选k是否等于1,不等于1要计算，如果等于1不需要计算

             if self.k!=1:
                 #计算要等多长时间要进行下一次同步
                 self.calc_wait_time()
                 # 在这个地方开始计时，时间过去self.wait_time后开始下一次同步
                 self.record_syn_start = round(time.time(), 3)

             #现在将master置为空闲状态
             self.master_is_free=True


    #供ps通知master完成了梯度聚合的任务
    def ps_notice_master_pull_data_finished(self, request, context):
        print(request.notice)
        #当ps发送已经完成聚合梯度的时候，这时将is_ps_aggrate_finished=True,让其通知worker再次训练
        self.is_ps_aggrate_finished=True
        #  在这个地方算下一次的时间应该最合适
        return tf_pb2.status(code=200,mes="master收到了ps完成数据pull")


    def calc_wait_time(self):
        #  首先对不参与这一次同步的worker的剩余完成时间进行更新，不在self.temp_ip中的不用更新
        #通过all_worker_ip与self.tmp_ip求差集

        need_update_worker_ip=list(set(self.all_worker_ip)-set(self.tmp_ip))
        print("need_update:"+str(need_update_worker_ip))
        for need_ip in need_update_worker_ip:
            if (self.time_next[need_ip]-self.wait_syn_time)<0:
                self.time_next[need_ip]=round(random.uniform(0,2),4)
            else:
                self.time_next[need_ip] =round(self.time_next[need_ip]-self.wait_syn_time,3)

        print("cal" + str(self.time_next))

        time_data=list(self.time_next.values())


        self.wait_syn_time=select_wait(time_data,self.k)+0.1
        print("wait_time="+str(self.wait_syn_time))

        #完成下一次等待时间的计算






