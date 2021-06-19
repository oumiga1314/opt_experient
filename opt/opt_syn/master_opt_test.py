import grpc
import time
from utils import tf_pb2
from concurrent import futures
from utils import Master as stf
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT = '1998'
MAX_MESSAGE_LENGTH=600*1024*1024
def serve():

    #创建一个Server
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=200),options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), (
    'grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])

    #将master注册到该grpcServer上
    a=stf.register_master(grpcServer)

    #绑定端口
    grpcServer.add_insecure_port('[::]:{}'.format(_PORT))
    #运行服务
    grpcServer.start()
    print("start server at port {} ...".format(_PORT))
    try:
        while True:
            if a.is_worker_finish:
                a.is_worker_finish = False
                a.master_notice_ps()

            time.sleep(0.03)
            if a.is_ps_aggrate_finished:
                a.is_ps_aggrate_finished=False
                a.notice_worker_again_train()

            time.sleep(0.03)
            if a.k!=1:
                if a.master_is_free:
                   if len(a.finish_ip)!=0:
                    syn_end_time = round(time.time(), 3)
                    #防止等待时间过长，所有worker都已经到达
                    if(a.wait_syn_time<=(syn_end_time-a.record_syn_start))|(len(a.finish_ip)==a.worker_num):
                        a.master_is_free = False
                        #self.count = 0
                        a.tmp_ip = a.finish_ip
                        a.finish_ip = []
                        # 向ps通知，通知ps聚合梯度，并且把要聚合的worker的ip带过去
                        res = a.ps.master_notice_ps_to_pull_data_from_worker(tf_pb2.worker_fin_ip(w_ip=a.tmp_ip))
                        print(res.mes)




    except KeyboardInterrupt:
        grpcServer.stop(0)

if __name__=='__main__':
    serve()