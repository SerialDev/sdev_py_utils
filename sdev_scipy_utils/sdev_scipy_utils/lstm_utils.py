"""Numpy utility functions"""
import numpy as np
import pandas as pd
import array
import os
import bz2



def pd_time_matrix_np(data):
    """
    * type-def ::pd.df :: np.array]
    * ---------------{Function}---------------
    * Transform a 2D pd dataframe into a 3D matrix for use in LSTMs . . .
    * ----------------{Params}----------------
    * : data in a pandas dataframe
    * ----------------{Returns}---------------
    * np.matrix of shape (x, y, z) . . .
    """
    np_data = data.as_matrix()
    return np_data.reshape((1, data.shape[0], data.shape[1]))

def np_time_matrix_pd(data):
    """
    * type-def ::np.array -> pd.df
    * ---------------{Function}---------------
    * Transform a 3D matrix  into a 2D pd dataframe  for use in testing . . .
    * ----------------{Params}----------------
    * : data in a 3D numpy array
    * ----------------{Returns}---------------
    * 2D pandas dataframe . . .
    """
    return pd.DataFrame(data.reshape(data.shape[1], data.shape[2]))


def np_generate_data(data, time_steps, separate=True):
    """
    * Function: Generates data for LSTM  [40x faster]
    * Usage: np_generate_data(data, time_steps, separate=True) . . .
    * -------------------------------
    * This function returns a dictionary with train, validation and, test data
    * Separate flag returns X and Y if set to False
    """
    if separate:
        train_x, val_x, test_x = np_prepare_data(data, time_steps)
        train_x = train_x.astype('int32')
        val_x = val_x.astype('int32')
        test_x = test_x.astype('int32')
        return dict(train=train_x, val=val_x, test=test_x)
    else:
        train_x, val_x, test_x = np_prepare_data(data, time_steps)
        train_y, val_y, test_y = np_prepare_data(data, time_steps, labels=True)
        return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def np_prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    * Function: Given the number of `time_steps` and some data,
    * prepares training, validation and test data for an lstm cell.
    * Usage: np_prepare_data(data, time_steps, labels=labels, val_size, test_size . . .
    * -------------------------------
    * This function returns three np arrays of shape (0, 0, 0)
    * one for train, val and test data [40x faster than df version]
    """
    train, val, test = np_split_data(data, val_size, test_size)
    return (np_rnn_data(train, time_steps, labels=labels),
           np_rnn_data(val, time_steps, labels=labels),
           np_rnn_data(test, time_steps, labels=labels))

def np_split_data(data, val_size=0.1, test_size=0.1):
    """
    * Function: splits data to training, validation and testing parts
    * Usage: split_data(data, val_size, test_size) . . .
    * -------------------------------
    * This function returns a np array for training, validation and, test parts
    *
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data[:ntest]) * (1 - val_size)))
    train, val, test = data[:nval], data[nval:ntest], data[ntest:]
    return train, val, test


def np_rnn_data(data, time_steps, labels=False):
    """
    * Function: creates new np array based on previous observation
    * Usage: np_rnn_data(data, time_steps, labels=labels) . . .
    * -------------------------------
    * This function returns a new np array based on previous observations
    * A = [1, 2, 3, 4, 5]
    * time_steps = 2
    * -> labels = False[[1, 2], [2, 3], [3,4]]
    * -> labels = True[2, 3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data[i + time_steps])
            except AttributeError:
                print(AttributeError)
        data_ = data[i: i + time_steps]
        rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data])
    return np.array(rnn_df)

def split_data(data, val_size=0.1, test_size=0.1):
     """
     splits data to training, validation and testing parts
     """
     ntest = int(round(len(data) * (1 - test_size)))
     nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

     df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

     return df_train, df_val, df_test

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
   """
   Given the number of `time_steps` and some data,
   prepares training, validation and test data for an lstm cell.
   """
   df_train, df_val, df_test = split_data(data, val_size, test_size)
   return (rnn_data(df_train, time_steps, labels=labels),
           rnn_data(df_val, time_steps, labels=labels),
           rnn_data(df_test, time_steps, labels=labels))

def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [2, 3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                print(AttributeError)
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)

def vectorized_lstm_data(data):
    return data.reshape(1, data.shape[0], data.shape[1])

def vectorized_lstm_feature(classes_array, feature_array, axis=1):
    def classes_ret(number):
        try:
            return (classes_array == number).astype(int)
        except AttributeError:
            return (classes_array == number)
    return np.apply_along_axis(classes_ret, axis, feature_array)

def batch_lstm_features(classes_array, feature_array, axis=1):
    features = np.empty((1))
    feature_array = feature_array.reshape(1, feature_array.shape[0])
    batches = feature_array.shape[0]//1000
    for num, tbatches in enumerate(batches, start=1):
        tmp = vectorized_lstm_feature(classes_array, feature_array[(tbatches * 1000):(num * 1000)])
        np.concatenate((features, tmp))
    features = features[1:]
    return features

import pickle

def to_sarvam(data, save_path='', save_file=False):
    with open("tmp.plk", 'wb') as tmp:
        pickle.dump(data, tmp)

        path = tmp.name
        ln = os.path.getsize(path)

    width = 128

    rem = ln % width

    a = array.array("B")

    with open("tmp.plk", 'rb') as tmp:
        a.fromfile(tmp, ln-rem)

    g = np.reshape(a, (len(a)//width, width))
    g = np.uint8(g)
    if save_file:
        file_format = 'png'
        file_path = save_path
        file_flag = os.path.exists('{}.{}'.format(file_path, file_format))
        #if file_flag:
        #    file_path = unique_filename_int(file_path)
        #scipy.misc.imsave(r'{}\\.{}'.format(file_path, file_format), g)
        scipy.misc.imsave(save_path, g)
    else:
        return g

def get_compressed_file(data):
    """
    * Function: compress data and get the compressed and uncompressed size
    * Usage: get_compressed_file(data) . . .
    * -------------------------------
    * This function returns
    * size_compressed, size_uncompressed
    """
    with bz2.BZ2File('tmp.bzip', 'wb') as tmp:
        pickle.dump(data, tmp)
    #compressed = open('tmp.bzip')
    size_compressed = os.path.getsize('tmp.bzip')
    with open('tmp.plk', 'wb') as tmp:
        pickle.dump(data, tmp)
    size_uncompressed = os.path.getsize('tmp.plk')
    return size_compressed, size_uncompressed


def get_binary_features(df, col, batches=False, batches_size=100000):
    classes_array = df[col].unique()
    series_np = df[col].as_matrix()
    series_np = series_np.reshape(series_np.shape[0], 1)
    if batches=='False':
        binary_descriptions = vectorized_lstm_feature(classes_array, series_np)
        return binary_descriptions
    else:
        batches = df[col].shape[0]//batches_size
        for finish, start in enumerate(range(batches),start=1):
            finish = finish * batches_size
            start = start * batches_size
            if start == 0:
                binary_descriptions = vectorized_lstm_feature(classes_array, series_np[start:finish])
            tmp = vectorized_lstm_feature(classes_array, series_np[start:finish])
            try:
                binary_descriptions = np.concatenate((binary_descriptions, tmp), axis=0)
            except MemoryError as e:
                print(e)
                print(start)
                return binary_descriptions
        return binary_descriptions

#def conv2d(x, W):
#    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    For KERAS
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * keras.layers.K.elu(x, alpha)
