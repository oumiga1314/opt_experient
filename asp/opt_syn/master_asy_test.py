import grpc
import time
from concurrent import futures
from utils import Master_asy as stf
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

            time.sleep(_ONE_DAY_IN_SECONDS)

    except KeyboardInterrupt:
        grpcServer.stop(0)

if __name__=='__main__':
    serve()