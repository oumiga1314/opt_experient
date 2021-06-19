
from utils.utils import *
from utils import NeuralNetwork as nn
from utils import tf_pb2_grpc,tf_pb2
import time

def register_parameterServer(grpc_server):
    a=ParameterServer()
    tf_pb2_grpc.add_psServicer_to_server(a, grpc_server)
    return a
class ParameterServer(tf_pb2_grpc.psServicer):
    def __init__(self):
        self.workers = {} #用来存储同各个worker的连接
        self.master=None #用来存储对master的连接通道
        self.net=None
        self.parms={}
        self.is_ready_start=False   #用来存储是否要从worker那取参数的信号
        self.is_ready_aggrate=False #用来存储是否要聚合参数的信号
        self.gridents_w = None  #用来临时存储权重
        self.gridents_b = None  #y用来临时存储biase
        self.w = None
        self.b = None
        self.shape=None #用来存储网络的层数，每层有多少个神经元
        self.count=0 #用来存储此次同步有多少个worker,好对参数进行取均值
        self.learn_rate=None#代表学习速率
        self.finished_ip=[], #用来记录已经完成任务的ip，也就是哪些节点完成任务
        self.temped_ip=[],#用来记录暂时完成的ip
        self.count_total=0#为了改变学习率




    def register_cluster(self, request, context):
        #如果不强转list,则运行失败，因为在NeuralNetwork.py初始化是判断shape类型必须是list
        self.finished_ip=[]
        self.temped_ip=[]
        self.shape=list(request.layer)
        self.net=nn.NeuralNetwork(self.shape, activation=nn.sgm)
        self.learn_rate=request.learn_rate
        self.master=connect_master(request.master)

        self.batch_size=request.batch_size
        #完成第一次的参数初始化
        self.w=self.net.get_weight()
        self.b=self.net.get_b()

        self.parms['weight']=tf_pb2.delta(w=to_data_one_row(self.w),b=to_data_one_row(self.b))

        for ws_addr in request.workers:
            key_ip =ws_addr.split(":")[0]
            #以键值对的方式存储
            self.workers[key_ip]=connect_worker(ws_addr)
            #在这个地方给threshold赋值,可以在这个地方调整系数，改变候选k的阈值
            self.threshold=len(self.workers)*0.5
        return tf_pb2.status(code=200, mes="ps register cluster success")

     #把参数发给各个worker
    def send_weight(self, request, context):
        return self.parms['weight']


    def master_notice_ps_to_pull_data_from_worker(self, request, context):
        self.finished_ip=request.w_ip
        self.is_ready_aggrate=True
        print(self.finished_ip)
        return tf_pb2.status(code=200,mes="ps收到了master聚合通知的任务")

    def master_notice_ps_aggregate(self, request, context):
        self.is_ready_aggrate=True
        return  tf_pb2.status(code=200,mes="ps收到聚合的信息")

    def aggregate_gradient(self):
         start_aggrate=round(time.time(),3)
         self.is_ready_aggrate=False
         self.temped_ip=self.finished_ip
         self.finished_ip=[]


         num_workers = len(self.temped_ip)#用来记录本次有多台worker要进行梯度聚合
         gridents_w = [0] * num_workers
         gridents_b = [0] * num_workers



         #把参数从参与本次梯度聚合的worker中把参数取回
         bian=0
         for i in self.temped_ip:
            if self.count_total%600==0:
                 self.learn_rate=self.learn_rate-0.002;
            self.count_total=self.count_total+1
            response = self.workers[i].send_delta.future(
                tf_pb2.status(code=200, mes="ps向worker" + str(i) + "请求梯度")).result()
            # with open("3.txt","a") as f:
            #     f.write(str("#####"))
            #     f.write(str(one_row_to_data(response.w, self.shape)))
            #     f.write(str(one_row_to_b(response.b, self.shape)))
            gridents_w[bian] = one_row_to_data(response.w, self.shape)
            gridents_b[bian] = one_row_to_b(response.b, self.shape)
            bian = bian + 1

        # 把来自给个worker的梯度相加，确认这段代码没有问题
         total_w, total_b = gridents_w[0], gridents_b[0]
         for j in range(num_workers - 1):
            total_w = np.add(total_w, gridents_w[j + 1])
            total_b = np.add(total_b, gridents_b[j + 1])

        # 正式更新梯度
            # print(total_w)

         self.w -= np.array(total_w) * self.learn_rate /(num_workers*self.batch_size)
         self.b -= np.array(total_b) * self.learn_rate / ( num_workers*self.batch_size)
         self.parms['weight'] = tf_pb2.delta(w=to_data_one_row(self.w), b=to_data_one_row(self.b))
         print("梯度聚合成功")
         # with open("1.txt", "a") as f:
         #     f.write(str("#####"))
         #     f.write(str(self.w))
         #     f.write(str(self.b))
         end_aggrate=round(time.time(),3)
         #聚合成功后给master发送一个通知,并告诉master这次聚合数据需要的时间

         aggrate_time = round(end_aggrate-start_aggrate,3)
         print(aggrate_time)

         self.master.ps_notice_master_pull_data_finished(tf_pb2.note(notice="ps完成了本次梯度聚合的任务"))





