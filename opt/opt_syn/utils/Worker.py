import time
import numpy as np
np.set_printoptions(threshold=np.inf)
from data import load_cifar10
from utils.BackProgation import backProgation
from utils import tf_pb2_grpc
from utils.utils import *
import csv
from utils import Worker

def register_worker(grpc_server):
    a = Worker()
    tf_pb2_grpc.add_workerServicer_to_server(a, grpc_server)
    return a
class Worker(tf_pb2_grpc.workerServicer):
    def __init__(self):
        self.ps = None #用于连接对ps的通道
        self.master=None #用于连接对master的通道
        self.train_x=None
        self.train_y=None
        self.delta_=None,#用于存储要发送给ps的delta
        self.bias_=None,#用来存储要发送给ps的bias
        self.layer=[]
        self.count=None #用来记录本worker训练了数据多少次，
        self.batch_size=None #用来记录每个批次训练数据的大小
        self.num_iter=None,#用来记录完成在本地训练数据的迭代次数，还要慎重思考
        self.test_x=None,
        self.test_y=None
        self.result=None,#代表对下一次迭代时间的预测
        self.ip_local=None,#代表本地的ip地址，在通知ps完成任务的时候，携带自己的ip
        self.read_=None,#用来读500条迭代时间的数据
        self.is_ready_start=False
        self.count_store=0 #为了存储训练过程的精确度，cost,时间而特意定的一个全局变量




    def register_cluster(self, request, context):
        ps_addr = request.ps
        master_addr=request.master
        self.layer=request.layer
        print(request.batch_size)
        self.batch_size=request.batch_size
        self.ip_local=get_host_ip()
        self.ps =connect_ps(ps_addr)

        self.master=connect_master(master_addr)
        self.out=None

        self.count=0
        self.read_=1
        self.read_go=100



        return tf_pb2.status(code=200, mes="worker register cluster succes")

    #有master调用将数据发送给没给worker

    def send_train_data(self, request, context):#由master负责发送该worker应该获得哪些数据，然后从本地加载数据（这是为了方便）
         index = request.index
         data_size = request.index_gap
         print(index)
         print(data_size)
         #通过索引位置和数据的大小来确定worker要训练多少条数据
         #self.x_tain和self.y_trian是为了测试训练集的精确度和损失函数
         #self.test_x和self.test_y是为了测试测试卷的精确度和损失函数
         (self.x_train,self.y_train), (self.test_x,self.test_y)=load_cifar10.load_cifar("../opt_syn/data/cifar-10-batches-py")

        #x_train是为了取在这个worker训练的数据
         self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1])/255
        #除以255是为了做归一化处理，对测试数据集进行处理
         self.test_x=self.test_x.reshape(self.test_x.shape[0],self.test_x.shape[1])/255
         self.test_y=self.test_y

         #对训练集数据进行处理
         self.train_x=self.x_train[index*data_size:(index+1)*data_size,:]
         self.train_y=self.y_train[index*data_size:(index+1)*data_size]


         # 判断如何分配数据,这些地方应该还可以优化
         self.num_iter = int(self.train_x.shape[0] / (self.batch_size))
         if (self.train_x.shape[0] % self.batch_size) != 0:
             self.num_iter = self.num_iter + 1

         return tf_pb2.status(code=200,mes="worker received data")

    #正式进行开始训练的函数
    def start_train(self):
            self.is_ready_start=False
            #记录开始训练的起始时间
            start_time=time.time()
            #先把ps的参数给请求过来
            response = self.ps.send_weight(tf_pb2.step(step_number=10))
            weights = one_row_to_data(response.w, self.layer)
            bias = one_row_to_b(response.b, self.layer)
            #对worker端的计算进行初始化
            bp = backProgation(self.layer)
            #这一步去取这一次准备训练的数据
            if self.count<self.num_iter-1:
                batch_x = self.train_x[self.count*self.batch_size:(self.count+1)*self.batch_size, :]
                batch_y = self.train_y[self.count*self.batch_size:(self.count+1)*self.batch_size]

            else:
                batch_x = self.train_x[self.count * self.batch_size:, :]
                batch_y = self.train_y[self.count * self.batch_size:]

            #训练数据确定后，将self.count+1
            self.count=self.count+1
            self.batch_size=batch_x.shape[0]
            print(self.batch_size)
            delta_, bias_ = bp.train(train_data=batch_x, train_labels=batch_y,
                                     batch_size=self.batch_size, weights=weights, bias=bias)


            #先把计算好的要发送的梯度放在本地
            self.delta_=to_data_one_row(delta_)
            self.bias_=to_data_one_row(bias_)
             #在这个地方去调用测试集的目的是为了保持每次迭代的时间基本一样，不会由于要测试，造成某次的时间特别长
            train_acc, train_cost = bp.predict(self.x_train, self.y_train, weights, bias)
            print("\t train acc the train set is {0:.2f}%\n".format(train_acc * 100))
            print("\t train  cost the train set is:" + str(round(train_cost, 4)))

            test_acc, test_cost = bp.predict(self.test_x, self.test_y, weights, bias)
            print("\t test acc the test set is {0:.2f}%\n".format(test_acc * 100))
            print("\t test  cost the test set is:" + str(round(test_cost, 4)))
            write_to_excel(self.count_store, round(time.time(), 4),
                           round(train_acc, 4),
                           round(train_cost, 4),
                           round(test_acc, 4),
                           round(test_cost, 4)
                           )

            #这地方是为了在训练完成一个epoch后测试下，该轮迭代的损失函数，所以我需要判断是否已经完成一个epoch
            if self.count==self.num_iter:
               self.count=0
               # 只记录一次完整训练完所有的数据，然后在保存
            # self.count=self.count+1
            self.count_store = self.count_store + 1
            #time.sleep(7)
            end_time =time.time()
            result_ = round((end_time - start_time), 3)
            print("time=" + str(result_) + ":" + str(self.count))

            #现在这个地方提取500条迭代实验的数据
            cpu_,store_=get_cpu_store()
            # 1. 创建文件对象
            # if self.read_<= self.read_go:
            #     self.read_=self.read_+1
            #     store_time(round(cpu_/100,3),round(store_/100,3),result_)

            #在这个地方还应该有一个使用NARX网络来预测下一次的迭代时间
            #目前现在对下一次迭代时间给了一个定值
            self.result=result_
            print("worker finished train")

            #完成训练后，通知下master自己完成了训练
            self.master.worker_notice_master_finished(tf_pb2.worker_note(w_ip=self.ip_local,
            w_f=str(self.ip_local)+"worker finished train",next_time=self.result))



     #定义一个由ps来提取worker计算梯度的函数
    def send_delta(self, request, context):
        print(request.mes)
        return tf_pb2.delta(w=self.delta_,b=self.bias_)


    #提供master通知自己开始训练的的方法
    def notice_start_train(self, request, context):
        print(request.notice)

        # self.start_train()
        self.is_ready_start=True

        return tf_pb2.status(code=200,mes="worker完成的训练")




    def ff(self):
        if self.is_ready_start==True:
            self.start_train()
