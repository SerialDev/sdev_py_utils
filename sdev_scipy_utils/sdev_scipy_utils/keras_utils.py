from keras import callbacks
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def coeff_determination(y_true, y_pred):
    from keras import backend as K

    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())


def tensorboard_callback(log_dir="./Graph"):
    return callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True
    )


def early_stopping(monitor="loss"):
    return callbacks.EarlyStopping(monitor=monitor, patience=10, verbose=1, mode="auto")


def model_with_weights(model, weights, skip_mismatch):
    """Load weights for model.

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


class Batcher:
    # Usage
    # net = net.train()  # explicitly set
    # bat_size = 40
    # loss_func = T.nn.MSELoss()
    # optimizer = T.optim.Adam(net.parameters(), lr=0.01)
    # batcher = Batcher(num_items=len(norm_x),
    #   batch_size=bat_size, seed=1)
    # max_epochs = 100
    def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  # ugh.
        else:
            result = self.indices[self.ptr : self.ptr + self.batch_size]
            self.ptr += self.batch_size
            return result
