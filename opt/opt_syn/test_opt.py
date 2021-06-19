


cluster ={
        "master":"192.168.1.104:2222",
        "workers":[
                    # "192.168.1.104:2223",
                    # "192.168.1.105:2225",
                    # "192.168.1.106:2226",
                    # "192.168.1.115:2227",
                    # "192.168.1.110:2228",
                    # "192.168.1.111:2229",
                    # "192.168.1.112:2230",
                    # "192.168.1.113:2231",
                    #"192.168.1.147:2227",
                    # "192.168.1.148:2228",
                    # "192.168.1.149:2229",
                    # "192.168.1.150:2230",
                    # "192.168.1.152:2231",
                   ],
                  "ps":"192.168.1.104:2224",
        "layer":[784,200,10],
        "learn_rate":0.03,
        "batch_size":128
    }
from utils.Client import Client
client = Client()
#开始注册集群相关信息
client.register_cluster(cluster)
#发送开始训练的通知给master
client.train()
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
import time
while True:
    time.sleep(_ONE_DAY_IN_SECONDS)



