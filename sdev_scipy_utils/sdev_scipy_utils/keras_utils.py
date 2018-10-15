from keras import callbacks
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true-y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def tensorboard_callback(log_dir='./Graph'):
    return callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)


def early_stopping(monitor='loss'):
    return callbacks.EarlyStopping(monitor=monitor, patience=10, verbose=1, mode='auto')


def model_with_weights(model, weights, skip_mismatch):
    """ Load weights for model.

    Args
    model : The model to load weights for.
    weights : The weights to load.
    skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.
    """
    if weights is not None:
        model.load_weights(weights, by_name=True, skip_mismatch=skip_mismatch)
    return model


def print_trainable_layers(model, trainable=True):
    for idx, layer in enumerate(model.layers):
        if trainable:
            if layer.trainable:
                print(idx, layer.trainable, layer)
        else:
            if not layer.trainable:
                print(idx, layer.trainable, layer)
    return model


def freeze_trainable_layers(model, except_last=4):
    for layer in model.layers[:-except_last]:
        layer.trainable = False
    return model
