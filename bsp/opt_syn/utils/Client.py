import grpc
from utils.utils import *
from .utils import tf_pb2

class Client(object):

    def __init__(self):
        self.master=None
    def register_cluster(self,cluster_info):

        #得到一个通道
        self.master= connect_master(cluster_info['master'])
        try:
            status = self.master.register_cluster(
            tf_pb2.cluster(ps=cluster_info['ps'], workers=cluster_info['workers'],
                           master=cluster_info['master'],
                           layer=cluster_info["layer"],learn_rate=cluster_info['learn_rate'],batch_size=cluster_info['batch_size'])
            )
            print(status.mes)


        except grpc._channel._Rendezvous as e:
            print("master服务异常，请检查")
            exit(0)


    def train(self):
        try:
            response =self.master.notice_train(tf_pb2.note(notice="client 通知master开始训练"))
            print(response.mes)
        except KeyboardInterrupt:
            print("master 服务器异常，请检查")
            exit(0)


