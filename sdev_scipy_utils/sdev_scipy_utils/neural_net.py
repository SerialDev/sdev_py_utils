import tensorflow as tf

# Input: 29 -> encode: 100 -> latent -> decode: 100 -> Output: 29.


class AE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(AE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(None, 118), name="InputLayer"),
                tf.keras.layers.Dense(
                    100,
                    kernel_initializer="uniform",
                    activation="tanh",
                    name="Encoder_1",
                ),
                tf.keras.layers.Dense(
                    latent_dim,
                    kernel_initializer="uniform",
                    activation="tanh",
                    name="Laten_Space",
                ),
            ],
            name="Encoder",
        )

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(
                    100, kernel_initializer="uniform", activation="tanh"
                ),
                tf.keras.layers.Dense(
                    118, kernel_initializer="uniform", activation="linear"
                ),
            ],
            name="Decoder",
        )
        self.AE_model = tf.keras.Model(
            inputs=self.encoder.input,
            outputs=self.decoder(self.encoder.output),
            name="Auto Encoder",
        )

    def call(self, input_tensor):
        latent_space = self.encoder.output
        reconstruction = self.decoder(latent_space)
        AE_model = tf.keras.Model(inputs=self.encoder.input, outputs=reconstruction)
        return AE_model(input_tensor)

    def summary(self):
        return self.AE_model.summary()


def mse_reconstruction(model, X_train):
    # Mean squared error, identify the reconstruction error in order to detect anomalies
    train_mse = np.mean(np.power(model(X_train) - X_train, 2), axis=1)
    train_error = pd.DataFrame(
        {"Reconstruction_error": train_mse, "True_class": y_train}
    )
    return train_error


def AE_predictor(X, model, threshold):
    """
    if mse > threshold, then we can classify this as an anomaly
    """
    X_valid = model(X)
    mse = np.mean(np.power(X_valid - X, 2), axis=1)
    y = np.zeros(shape=mse.shape)
    y[mse > threshold] = 1
    return y


def plot_confusion_matrix(y_test, y_pred):
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred))

    plt.figure()
    sns.heatmap(cm, cmap="coolwarm", annot=True, linewidths=0.5)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Real class")
    plt.show()


#

import torch
import math
import numpy as np
import torch.optim as optim
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from collections import defaultdict


def set_device(gpu=-1):
    if gpu != -1 and torch.cuda.is_available():
        device = torch.device("cuda: " + str(gpu))
    else:
        device = torch.device("cpu")
    return device


def umap_reshape(data, enclose=False):
    c_shape = data.shape
    result = []
    for i in range(len(c_shape)):
        if i == 0 and c_shape[i] > 1:
            result.append(c_shape[i])
        else:
            pass
        if i > 0:
            result.append(c_shape[i])
    final_result = 1
    for i in range(len(result)):
        final_result = final_result * result[i]
    if enclose:
        return data.reshape(1, final_result)
    else:
        return data.reshape(final_result)


def max_len_pad(df, _dtype=np.int32):
    import numpy as np

    max_len = max(map(len, df))

    result_df = df.apply(
        lambda x: np.concatenate([x, np.zeros((max_len - x.shape[0]), dtype=_dtype)])
    )

    return result_df


def encode_tokenize(df, col, tokenizer):
    import torch

    digits = df[col].fillna("nn").apply(lambda x: torch.tensor([tokenizer.encode(x)]))
    return digits


def embed_tokens_bert(digits, bert_model):
    import torch

    digits = digits.apply(lambda x: umap_reshape(bert_model(x)[0].detach().numpy()))
    return digits


def clean_text(text):
    import re

    text = re.sub("[^A-Za-z0-9]+", " ", text)
    return text


def np_topk(data, k):
    return data.argsort()[-k:][::-1]


def x_only_train_test_split(data, test_size=0.33):

    from sklearn.model_selection import train_test_split

    X_train, X_test, _, _ = train_test_split(
        data, np.zeros(data.shape[0]), test_size=test_size, random_state=42
    )
    return X_train, X_test


def pd_get_dummies_concat(source_df, column, prefix=None, drop=True):
    result = pd.concat(
        [source_df, pd.get_dummies(source_df[column], prefix=prefix)],
        axis=1,
        sort=False,
    )
    if drop:
        result.drop(column, axis=1, inplace=True)
    return result


def pd_timed_slice(
    df,
    timeseries,
    week=0,
    day=0,
    hour=0,
    minute=0,
    second=0,
    millisecond=0,
    microsecond=0,
):
    import datetime

    df[timeseries] = pd.to_datetime(df[timeseries])
    range_max = df[timeseries].max()
    range_min = range_max - datetime.timedelta(
        weeks=week,
        days=day,
        hours=hour,
        minutes=minute,
        seconds=second,
        milliseconds=millisecond,
        microseconds=microsecond,
    )
    sliced_df = df[(df[timeseries] >= range_min) & (df[timeseries] <= range_max)]
    return sliced_df
