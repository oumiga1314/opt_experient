import numpy as np
import os
np.set_printoptions(threshold=np.inf)

# 网站给出的读取CIFAR10的函数
def load_data(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        # print(dict.keys())
        # 可以看出这个字典的keys
        # print(dict.__contains__(b'data'))
    X = dict[b'data']
    Y = dict[b'labels']
    # 将（10000，3072）转变成为图片格式，其中转置函数是为了符合通道在最后的位置
    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
    Y = np.array(Y)
    return X, Y


# 输入文件目录，输出数据集的训练集数据和标签，测试集数据和标签
def load_cifar(ROOT):
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % b)
        X, Y = load_data(f)
        xs.append(X)
        ys.append(Y)
    # print(len(xs))
    Xtr = np.concatenate(xs)
    # print(Xtr.shape)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_data(os.path.join(ROOT, 'test_batch'))
    Xtr=Xtr.reshape(Xtr.shape[0],Xtr.shape[1]*Xtr.shape[2]*Xtr.shape[3])
    Xte = Xte.reshape(Xte.shape[0], Xte.shape[1] * Xte.shape[2] * Xte.shape[3])

    # permutation = np.random.permutation(Xtr.shape[0])
    shuffled_dataset = Xtr
    shuffled_labels =  Ytr
    shuffled_labels=shuffled_labels.reshape(Xtr.shape[0],1)
    Yte=Yte.reshape(Yte.shape[0],1)
    return (shuffled_dataset, shuffled_labels), (Xte, Yte)

#load_cifar('cifar-10-batches-py')