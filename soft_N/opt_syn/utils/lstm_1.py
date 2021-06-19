import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import concatenate
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def cs_to_sl():
    # load dataset
    dataset = pd.read_csv('data.csv')
    print(type(dataset))
    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[3,4]], axis=1, inplace=True)
    print(reframed)
    return reframed, scaler

cs_to_sl()
def cs_to_sl_1(data):
    # load dataset
    values = pd.DataFrame(data).values
    print(values)
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[3,4]], axis=1, inplace=True)
    print(reframed)
    return reframed, scaler

#cs_to_sl_1([[1,3,4]])

def train_test(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_hours = 200
    train = values[:n_train_hours, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    return train_X, train_y

def train_test_1(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_hours = 200
    train = values[263:264,:-1]

    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    return train_X, train_y

def fit_network(train_X, train_y):
    model = Sequential()
    model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=40,  verbose=2,
                        shuffle=False)
    return model

#
# def start_train():
#     train_X, train_y, test_X, test_y = train_test(reframed)
#     print(test_X[0:1, ])
#
#     model = fit_network(train_X, train_y, test_X, test_y, scaler)

# if __name__ == '__main__':
#     # drow_pollution()
#     reframed, scaler = cs_to_sl()
#     train_X, train_y = train_test(reframed)
#
#
#     model= fit_network(train_X, train_y)
#
#
#     # print(test_x.shape)
#
#     # make a prediction

