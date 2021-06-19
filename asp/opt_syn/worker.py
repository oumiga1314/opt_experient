import time
import grpc
from concurrent import futures
from utils import Worker as stf
_ONE_DAY_IN_SECONDS = 60 * 60 * 24*5
_PORT = '4003'
MAX_MESSAGE_LENGTH=600*1024*1024
def serve():
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=600),options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH), (
    'grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    a = stf.register_worker(grpcServer)

    grpcServer.add_insecure_port('[::]:{}'.format(_PORT))
    grpcServer.start()
    print("start server at port {} ...".format(_PORT))
    try:
        while True:
            if a.is_ready_start:
               a.ff()
            #stf.Worker.ff(stf.Worker())
            time.sleep(0.03)
    except KeyboardInterrupt:
        grpcServer.stop(0)


if __name__ == '__main__':
    serve()
