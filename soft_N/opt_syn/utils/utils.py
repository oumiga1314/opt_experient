import grpc
import numpy as np
import socket
from utils import tf_pb2, tf_pb2_grpc
import psutil
import csv
def connect_master(addr):
    conn = grpc.insecure_channel(addr)
    return tf_pb2_grpc.masterStub(channel=conn)


def connect_worker(addr):
    conn = grpc.insecure_channel(addr)
    return tf_pb2_grpc.workerStub(channel=conn)


def connect_ps(addr):
    conn = grpc.insecure_channel(addr)
    return tf_pb2_grpc.psStub(channel=conn)


def to_data_one_row(x):
    """
    weight,b都可以用这个函数处理
    :param x:原格式的数据
    :return: 变成一行的数据，
    """
    tmp = []
    for i in x:
        a = list(i.flatten())
        tmp = tmp + a
    return tmp


#将一行数据还原x带数据，layer代表每一层的神经元个数

def one_row_to_data(x,layer):
    """
    :param x:数据
    :param layer:各个层的神经元
    :return: 返回恢复原格式的数据
    """
    result = []
    start = 0
    end   = 0
    data = np.array(x)
    for i, j in zip(layer[:-1], layer[1:]):
        start = end
        end = end + i * j
        temp = data[start:end,]
        result.append(temp.reshape(j, i))

    return result

def one_row_to_b(x,layer):
    """
    :param x: 要转换的数据
    :param layer: 层数的相关信息
    :return: 返回处理好的数
    """
    result = []
    start = 0
    end = 0
    data = np.array(x)
    for i in layer[1:]:
        start = end
        end = end + i
        temp = data[start:end,]
        result.append(temp.reshape(i, 1))
    return  result

def get_host_ip():
    """
    查询本机ip地址
    :return:
    """
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(('8.8.8.8',80))
        ip=s.getsockname()[0]
    finally:
        s.close()
    return ip

def get_cpu_store():
    """
    查看当前的cpu和内存的使用情况
    :return: cpu的可使用百分比和内存的可使用率
    """
    store_ = psutil.virtual_memory().percent
    cup_ = psutil.cpu_percent(0.2)

    return 100-store_,100-cup_

def store_time(cpu_,store_,result_):
    f = open('../opt_syn/utils/data2.csv', 'a', encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    # # 3. 构建列表头
    # csv_writer.writerow(["cpu", "store", "time"])
    # 2. 基于文件对象构建 csv写入对象
    # 4. 写入csv文件内容
    csv_writer.writerow([cpu_, store_, result_])
    # 5. 关闭文件
    f.close()


from xlutils.copy import copy
import xlrd
import os
import xlwt
def write_to_excel(row,date_time,train_acc,train_cost,test_acc,test_cost):
    if (os.path.isfile("aa.xls"))!=True:
        """创建一个excel对象"""
        book = xlwt.Workbook(encoding='utf-8', style_compression=0)
        """创建sheet"""
        sheet = book.add_sheet('test', cell_overwrite_ok=False)
        book.save("aa.xls")
    workbook = xlrd.open_workbook("aa.xls")  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    #worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old =row # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    new_worksheet.write(rows_old, 0, row)
    new_worksheet.write(rows_old,1,date_time)
    new_worksheet.write(rows_old,2, train_acc)
    new_worksheet.write(rows_old,3, train_cost)
    new_worksheet.write(rows_old,4, test_acc)
    new_worksheet.write(rows_old,5, test_cost)
    new_workbook.save("aa.xls")  # 保存工作簿
