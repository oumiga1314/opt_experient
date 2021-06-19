import numpy as np
def load_data(path):
    path =path
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']

    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    y_train = y_train.reshape(y_train.shape[0], 1)

    y_test = y_test.reshape(y_test.shape[0], 1)


    return (x_train, y_train), (x_test, y_test)