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
        train_x = train_x.astype("int32")
        val_x = val_x.astype("int32")
        test_x = test_x.astype("int32")
        return dict(train=train_x, val=val_x, test=test_x)
    else:
        train_x, val_x, test_x = np_prepare_data(data, time_steps)
        train_y, val_y, test_y = np_prepare_data(data, time_steps, labels=True)
        return (
            dict(train=train_x, val=val_x, test=test_x),
            dict(train=train_y, val=val_y, test=test_y),
        )


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
    return (
        np_rnn_data(train, time_steps, labels=labels),
        np_rnn_data(val, time_steps, labels=labels),
        np_rnn_data(test, time_steps, labels=labels),
    )


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
        data_ = data[i : i + time_steps]
        rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data])
    return np.array(rnn_df)


def split_data(data, val_size=0.1, test_size=0.1):
    """
        * type-def ::[DataFrame] ::float ::float -> (DataFrame, DataFrame, DataFrame)
    * ---------------{Function}---------------
        * Splits input data into training, validation, and testing parts.
    * ----------------{Returns}---------------
        * -> df_train ::DataFrame | The training data portion
        * -> df_val   ::DataFrame | The validation data portion
        * -> df_test  ::DataFrame | The testing data portion
    * ----------------{Params}----------------
        * : data     ::DataFrame | The input data DataFrame
        * : val_size ::float | The proportion of data to be used for validation (default: 0.1)
        * : test_size::float | The proportion of data to be used for testing (default: 0.1)
    * ----------------{Usage}-----------------
        * >>> df_train, df_val, df_test = split_data(data, val_size=0.1, test_size=0.1)
        * >>> len(df_train), len(df_val), len(df_test)
        * (648, 81, 90)
    * ----------------{Notes}-----------------
        * This function assumes that the input data is a pandas DataFrame.
        * The remaining data (after subtracting validation and testing portions) will be used for training.
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = (
        data.iloc[:nval],
        data.iloc[nval:ntest],
        data.iloc[ntest:],
    )

    return df_train, df_val, df_test


def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
        * type-def ::[DataFrame] ::int ::Bool ::float ::float -> (np.ndarray, np.ndarray, np.ndarray)
    * ---------------{Function}---------------
        * Prepares input data for training, validation, and testing for an LSTM cell, given the number of time steps.
    * ----------------{Returns}---------------
        * -> train_data ::np.ndarray | The prepared training data
        * -> val_data   ::np.ndarray | The prepared validation data
        * -> test_data  ::np.ndarray | The prepared testing data
    * ----------------{Params}----------------
        * : data       ::DataFrame | The input data DataFrame
        * : time_steps ::int       | The number of time steps for the LSTM cell
        * : labels     ::Bool      | Whether to return the labels (default: False)
        * : val_size   ::float     | The proportion of data to be used for validation (default: 0.1)
        * : test_size  ::float     | The proportion of data to be used for testing (default: 0.1)
    * ----------------{Usage}-----------------
        * >>> train_data, val_data, test_data = prepare_data(data, time_steps=10, labels=False, val_size=0.1, test_size=0.1)
        * >>> train_data.shape, val_data.shape, test_data.shape
        * ((648, 10, 1), (81, 10, 1), (90, 10, 1))
    * ----------------{Notes}-----------------
        * This function assumes that the input data is a pandas DataFrame.
        * The function uses `split_data()` to split the data and `rnn_data()` to format the data for LSTM.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (
        rnn_data(df_train, time_steps, labels=labels),
        rnn_data(df_val, time_steps, labels=labels),
        rnn_data(df_test, time_steps, labels=labels),
    )


def generate_data(fct, x, time_steps, seperate=False):
    """
        * type-def ::Callable ::Any ::int ::Bool -> Tuple[Dict, Dict]
    * ---------------{Function}---------------
        * Generates data based on a function and prepares it for training, validation, and testing.
    * ----------------{Returns}---------------
        * -> x_data ::Dict | The prepared input data for train, val, and test
        * -> y_data ::Dict | The prepared output data for train, val, and test
    * ----------------{Params}----------------
        * : fct       ::Callable | The function to generate data
        * : x         ::Any      | The input to the function 'fct'
        * : time_steps::int      | The number of time steps for the LSTM cell
        * : seperate  ::Bool     | Whether to return separate columns for 'a' and 'b' (default: False)
    * ----------------{Usage}-----------------
        * >>> fct = lambda x: pd.DataFrame({"a": np.sin(x), "b": np.cos(x)})
        * >>> x = np.linspace(0, 10, 100)
        * >>> time_steps = 10
        * >>> generate_data(fct, x, time_steps, seperate=True)
    * ----------------{Notes}-----------------
        * This function generates data using a given function and prepares it for use with LSTM models.
    """
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data["a"] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(
        data["b"] if seperate else data, time_steps, labels=True
    )
    return (
        dict(train=train_x, val=val_x, test=test_x),
        dict(train=train_y, val=val_y, test=test_y),
    )


def rnn_data(data, time_steps, labels=False):
    """
        * type-def ::[DataFrame] ::int ::Bool -> np.ndarray
    * ---------------{Function}---------------
        * Creates a new DataFrame based on previous observations for use with an LSTM model.
    * ----------------{Returns}---------------
        * -> prepared_data ::np.ndarray | The prepared data in the format suitable for LSTM
    * ----------------{Params}----------------
        * : data       ::DataFrame | The input data DataFrame
        * : time_steps ::int       | The number of time steps for the LSTM cell
        * : labels     ::Bool      | Whether to return the labels (default: False)
    * ----------------{Usage}-----------------
        * >>> data = pd.DataFrame([1, 2, 3, 4, 5], columns=['values'])
        * >>> rnn_data(data, time_steps=2, labels=False)
        * array([[[1],
                  [2]],
                 [[2],
                  [3]],
                 [[3],
                  [4]]])
        * >>> rnn_data(data, time_steps=2, labels=True)
        * array([[2],
                 [3],
                 [4],
                 [5]])
    * ----------------{Notes}-----------------
        * This function assumes that the input data is a pandas DataFrame.
        * The function creates a new DataFrame with sequences of length 'time_steps' for use with LSTM models.
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
            data_ = data.iloc[i : i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])
    return np.array(rnn_df)


def vectorized_lstm_data(data):
    """
        * type-def ::np.ndarray -> np.ndarray
    * ---------------{Function}---------------
        * Reshapes the input data for use with an LSTM model.
    * ----------------{Returns}---------------
        * -> reshaped_data ::np.ndarray | The reshaped data in the format suitable for LSTM
    * ----------------{Params}----------------
        * : data ::np.ndarray | The input data array
    * ----------------{Usage}-----------------
        * >>> data = np.array([[1, 2], [3, 4], [5, 6]])
        * >>> vectorized_lstm_data(data)
        * array([[[1, 2],
                  [3, 4],
                  [5, 6]]])
    * ----------------{Notes}-----------------
        * The function reshapes the input data into a 3D array with shape (1, data.shape[0], data.shape[1]).
    """
    return data.reshape(1, data.shape[0], data.shape[1])


def vectorized_lstm_feature(classes_array, feature_array, axis=1):
    """
        * type-def ::np.ndarray ::np.ndarray ::int -> np.ndarray
    * ---------------{Function}---------------
        * Vectorizes LSTM feature based on the provided classes_array and feature_array.
    * ----------------{Returns}---------------
        * -> vectorized_feature ::np.ndarray | The vectorized feature array
    * ----------------{Params}----------------
        * : classes_array  ::np.ndarray | The classes array to compare against the feature_array
        * : feature_array  ::np.ndarray | The feature array to be vectorized
        * : axis           ::int        | The axis along which to apply the comparison (default: 1)
    * ----------------{Usage}-----------------
        * >>> classes_array = np.array([0, 1, 2])
        * >>> feature_array = np.array([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
        * >>> vectorized_lstm_feature(classes_array, feature_array)
        * array([[[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]],
                 [[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]],
                 [[0, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1]]])
    * ----------------{Notes}-----------------
        * The function vectorizes the LSTM feature based on the provided classes_array and feature_array.
    """
    def classes_ret(number):
        try:
            return (classes_array == number).astype(int)
        except AttributeError:
            return classes_array == number

    return np.apply_along_axis(classes_ret, axis, feature_array)


def batch_lstm_features(classes_array, feature_array, axis=1):
    """
        * type-def ::np.ndarray ::np.ndarray ::int -> np.ndarray
    * ---------------{Function}---------------
        * Creates batched LSTM features from the given classes_array and feature_array.
    * ----------------{Returns}---------------
        * -> batched_features ::np.ndarray | The batched LSTM features array
    * ----------------{Params}----------------
        * : classes_array  ::np.ndarray | The classes array to compare against the feature_array
        * : feature_array  ::np.ndarray | The feature array to be vectorized
        * : axis           ::int        | The axis along which to apply the comparison (default: 1)
    * ----------------{Usage}-----------------
        * >>> classes_array = np.array([0, 1, 2])
        * >>> feature_array = np.array([[0, 1, 2], [2, 1, 0], [1, 0, 2]])
        * >>> batch_lstm_features(classes_array, feature_array)
    * ----------------{Notes}-----------------
        * This function creates batched LSTM features by reshaping the input feature_array and
          vectorizing the LSTM features using the provided classes_array.
    """
    features = np.empty((1))
    feature_array = feature_array.reshape(1, feature_array.shape[0])
    batches = feature_array.shape[0] // 1000
    for num, tbatches in enumerate(batches, start=1):
        tmp = vectorized_lstm_feature(
            classes_array, feature_array[(tbatches * 1000) : (num * 1000)]
        )
        np.concatenate((features, tmp))
    features = features[1:]
    return features


import pickle


def to_sarvam(data, save_path="", save_file=False):
    """
    * type-def ::Any ::str ::bool -> np.ndarray | None
    * ---------------{Function}---------------
        * Converts the given data to a SARVAM (Scalable And Reusable Visual Analytical Models) representation.
    * ----------------{Returns}---------------
        * -> g ::np.ndarray | The SARVAM representation of the data (if save_file is False)
        * -> None           | None (if save_file is True)
    * ----------------{Params}----------------
        * : data      ::Any    | The data to be converted
        * : save_path ::str    | The path to save the SARVAM representation (default: "")
        * : save_file ::bool   | Whether to save the SARVAM representation as a file (default: False)
    * ----------------{Usage}-----------------
        * >>> data = np.array([1, 2, 3, 4, 5])
        * >>> sarvam_data = to_sarvam(data)
    * ----------------{Notes}-----------------
        * This function converts the given data to a SARVAM representation and saves it as a file if specified.
    """
    with open("tmp.plk", "wb") as tmp:
        pickle.dump(data, tmp)

        path = tmp.name
        ln = os.path.getsize(path)

    width = 128

    rem = ln % width

    a = array.array("B")

    with open("tmp.plk", "rb") as tmp:
        a.fromfile(tmp, ln - rem)

    g = np.reshape(a, (len(a) // width, width))
    g = np.uint8(g)
    if save_file:
        file_format = "png"
        file_path = save_path
        file_flag = os.path.exists("{}.{}".format(file_path, file_format))
        # if file_flag:
        #    file_path = unique_filename_int(file_path)
        # scipy.misc.imsave(r'{}\\.{}'.format(file_path, file_format), g)
        scipy.misc.imsave(save_path, g)
    else:
        return g


def get_compressed_file(data):
    """
    * type-def ::Any -> Tuple[int, int]
    * ---------------{Function}---------------
        * Compresses the given data and returns the compressed and uncompressed file sizes.
    * ----------------{Returns}---------------
        * -> size_compressed, size_uncompressed ::Tuple[int, int] | The compressed and uncompressed file sizes
    * ----------------{Params}----------------
        * : data ::Any | The data to be compressed
    * ----------------{Usage}-----------------
        * >>> data = np.array([1, 2, 3, 4, 5])
        * >>> size_compressed, size_uncompressed = get_compressed_file(data)
    * ----------------{Notes}-----------------
        * This function compresses the given data using the bz2 module and returns the compressed and uncompressed file sizes.
    """
    with bz2.BZ2File("tmp.bzip", "wb") as tmp:
        pickle.dump(data, tmp)
    # compressed = open('tmp.bzip')
    size_compressed = os.path.getsize("tmp.bzip")
    with open("tmp.plk", "wb") as tmp:
        pickle.dump(data, tmp)
    size_uncompressed = os.path.getsize("tmp.plk")
    return size_compressed, size_uncompressed


def get_binary_features(df, col, batches=False, batches_size=100000):
    """
    * type-def ::pd.DataFrame ::str ::Union[bool, str] ::int -> np.ndarray
    * ---------------{Function}---------------
        * Generates binary features from a DataFrame column.
    * ----------------{Returns}---------------
        * -> binary_descriptions ::np.ndarray | The binary features array
    * ----------------{Params}----------------
        * : df           ::pd.DataFrame       | The input DataFrame
        * : col          ::str                | The column name from which to generate binary features
        * : batches      ::Union[bool, str]   | Whether to process in batches (default: False)
        * : batches_size ::int                | The batch size if processing in batches (default: 100000)
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        * >>> binary_features = get_binary_features(df, "A")
    * ----------------{Notes}-----------------
        * This function generates binary features from the given DataFrame column using vectorized LSTM feature extraction.
    """
    classes_array = df[col].unique()
    series_np = df[col].as_matrix()
    series_np = series_np.reshape(series_np.shape[0], 1)
    if batches == "False":
        binary_descriptions = vectorized_lstm_feature(classes_array, series_np)
        return binary_descriptions
    else:
        batches = df[col].shape[0] // batches_size
        for finish, start in enumerate(range(batches), start=1):
            finish = finish * batches_size
            start = start * batches_size
            if start == 0:
                binary_descriptions = vectorized_lstm_feature(
                    classes_array, series_np[start:finish]
                )
            tmp = vectorized_lstm_feature(classes_array, series_np[start:finish])
            try:
                binary_descriptions = np.concatenate((binary_descriptions, tmp), axis=0)
            except MemoryError as e:
                print(e)
                print(start)
                return binary_descriptions
        return binary_descriptions


# def conv2d(x, W):
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
