import time
from concurrent import futures
import grpc
from utils import ParameterServer as stf
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_PORT = '1999'
MAX_MESSAGE_LENGTH=600*1024*1024
def serve():
    #创建一个参数服务器服务
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=1500),options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), (
    'grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])

    a=stf.register_parameterServer(grpcServer)

    grpcServer.add_insecure_port('[::]:{}'.format(_PORT))
    #开启服务
    grpcServer.start()

    print("start server at port {} ...".format(_PORT))

    try:
        while True:
            if a.is_ready_aggrate:
                a.aggregate_gradient()

    except KeyboardInterrupt:
        grpcServer.stop(0)

if __name__=='__main__':
    serve()