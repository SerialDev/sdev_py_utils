# --------{Object Serialization}--------#

import yaml
from collections import OrderedDict


def network_compress_encode(data):
    """
    * ---------------Function---------------
    * network_compress_encode(data)
    * ----------------Returns---------------
    * -> result ::str | b64 encoded data
    * ----------------Params----------------
    * data: serializable data | <any>
    * ----------------Usage----------------- Compress and encode data into a
    * base64 string. This function takes in a data as an input, compresses it using
    * zlib and then encodes it into a base64 string format. The compressed and
    * encoded data is returned as a string.

    """import zlib
    import base64

    result = base64.b64encode(zlib.compress(data.encode())).decode()
    return result


def network_compress_decode(data):
    """
    * --------------Function---------------
    *
    * Decodes and decompresses a network transmission
    *
    * ----------------Returns---------------
    *
    * -> result ::str |'Success' if the operation was successful, 'Failure'
    * otherwise
    *
    * ----------------Params----------------
    *
    * data ::<any> | The data to decode and decompress
    *
    * ----------------Usage-----------------
    *
    * This function is used to decode and decompress network data that has been
    *
    * encoded in base64 and compressed using zlib. The function takes in the
    *
    * encoded and compressed data as an argument and returns the decompressed
    *
    * data as a string. If there is an error in the decoding or decompression
    *
    * process, the function will return 'Failure'.
    *
    *
    * Example:
    *
    *
    * result = network_compress_decode('eJxNjYGsDA8kJUKr1Go5Iz8=')
    *
    *
    * In this example, the data being passed into the function is the base64
    * encoded
    *
    * and zlib compressed form of the string 'Hello, World!'. The function will
    *
    * decode and decompress the data, and return the original string. */ def
    * network_compress_decode(data): import zlib import base64
    *
    * try: result = zlib.decompress(base64.b64decode(data.encode())) return
    * 'Success' except: return 'Failure'

    """
    import zlib
    import base64

    result = zlib.decompress(base64.b64decode(data.encode()))
    return result


def to_buffer(data):
    """
    * ------------ Function ----------------
    * Converts a given data object into a BytesIO buffer using the `dill` module for serialization.
    * ----------------Returns-----------------
    * -> result :: io | io object if the operation was successful
    * ----------------Params----------------
    * data <any> - The object to be serialized and stored in the buffer.
    * ----------------Usage-----------------
    * to_buffer(data)
    *
    * Example:
    *
    * >>> to_buffer("Hello, World!")
    * <_io.BytesIO object at 0x104a67460>
    *
    * Note:
    * The function returns a BytesIO object that contains the serialized version of the input data.
    """
    import io
    import dill as pickle  # Dill allows serializing lambdas

    buf = io.BytesIO()
    buf.seek(0)
    buf.write(pickle.dumps(data))
    return buf


def from_buffer(buf):
    '''
    * ----------------Function---------------
    * from_buffer deserializes a byte buffer containing a pickled object.
    * ----------------Returns---------------
    * -> result :: <any> | The deserialized object.
    * ----------------Params----------------
    * buf :: <io.BytesIO> | A byte buffer containing a pickled object.
    * ----------------Usage-----------------
    * from_buffer(buf)
    *
    * Example:
    *
    * from_buffer(io.BytesIO(pickle.dumps({'foo': 'bar'})))
    '''
    import dill as pickle  # Dill allows serializing lambdas

    buf.seek(0)
    data = pickle.loads(buf.read())
    return data


def b64_pickle(data):
    '''
   * ---------------Function---------------
   * Encodes and pickles data using base64 and pickle modules
   * ----------------Returns---------------
   * -> result ::str | The base64 encoded pickled data if the operation was
   * successful,
   * 'Failure' otherwise
   * ----------------Params----------------
   * data ::<any>  | The data to be pickled and encoded
   * ----------------Usage-----------------
   * result = b64_pickle("some data")
   * if result != 'Failure':
        # do something with the result

    '''
    import base64
    import pickle

    return base64.b64encode(pickle.dumps(data))


def md5_hex(data):
    """
    * ---------------{Function}---------------
    * Calculates the MD5 hash of a byte string and returns its hexadecimal representation.
    * ----------------{Returns}---------------
    * -> hash_hex  ::str        |The hexadecimal representation of the MD5 hash
    * ----------------{Params}----------------
    * : data      ::bytes       |The byte string to hash
    * ----------------{Usage}-----------------
    * >>> md5_hex(b"hello world")
    * '5eb63bbbe01eeed093cb22bb8f5acdc3'
    """
    from hashlib import md5

    return md5(data).hexdigest()


def hash_runtime(data):
    """
    * ---------------{Function}---------------
    * Computes the MD5 hash of a Base64-encoded pickled object.
    * ----------------{Returns}---------------
    * -> hashed    ::str | The MD5 hash of the Base64-encoded pickled object
    * ----------------{Params}----------------
    * : data       ::Any | The data to be pickled and hashed
    * ----------------{Usage}-----------------
    * >>> hash_runtime("example string")
    * '78d699abb4e9b4f08e2f0a66eb23ecb5'
    * ----------------{Notes}-----------------
    * This function requires the `md5_hex()` and `b64_pickle()` functions to be defined elsewhere.
    * `md5_hex()` computes the MD5 hash of a string and returns the result as a hexadecimal string.
    * `b64_pickle()` pickles an object and returns the result as a Base64-encoded string.
    """
    return md5_hex(b64_pickle(data))


def load_yaml_odict(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    * Function: Load yaml into an OrderedDict
    * -----------{returns}------------
    * An OrderedDict object with yaml contents . . .
    * ------------{usage}-------------
    >>> load_yaml_odict(stream, yaml.SafeLoader)
    """

    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def dump_odict_yaml(data, stream=None, Dumper=yaml.Dumper, **kwds):
    """
    * Function: Dump an OrderedDict onto a yaml file
    * -----------{returns}------------
    * Serialised OrderedDict object into yaml . . .
    * ------------{usage}-------------
    >>> dump_odict_yaml(data, Dumper=yaml.SafeDumper)
    """

    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
        )

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


# -{Recently used finite size container}#


from collections import MutableMapping, OrderedDict
from threading import Lock


_Null = object()


# This object is maintained under the urllib3 codebase.
class RecentlyUsedContainer(MutableMapping):
    """
    Provides a thread-safe dict-like container which maintains up to
    ``maxsize`` keys while throwing away the least-recently-used keys beyond
    ``maxsize``.
    :param maxsize:
        Maximum number of recent elements to retain.
    :param dispose_func:
        Every time an item is evicted from the container,
        ``dispose_func(value)`` is called.  Callback which will get called
    """

    ContainerCls = OrderedDict

    def __init__(self, maxsize=10, dispose_func=None):
        self._maxsize = maxsize
        self.dispose_func = dispose_func

        self._container = self.ContainerCls()
        self._lock = Lock()

    def __getitem__(self, key):
        # Re-insert the item, moving it to the end of the eviction line.
        with self._lock:
            item = self._container.pop(key)
            self._container[key] = item
            return item

    def __setitem__(self, key, value):
        evicted_value = _Null
        with self._lock:
            # Possibly evict the existing value of 'key'
            evicted_value = self._container.get(key, _Null)
            self._container[key] = value

            # If we didn't evict an existing value, we might have to evict the
            # least recently used item from the beginning of the container.
            if len(self._container) > self._maxsize:
                _key, evicted_value = self._container.popitem(last=False)

        if self.dispose_func and evicted_value is not _Null:
            self.dispose_func(evicted_value)

    def __delitem__(self, key):
        with self._lock:
            value = self._container.pop(key)

        if self.dispose_func:
            self.dispose_func(value)

    def __len__(self):
        with self._lock:
            return len(self._container)

    def __iter__(self):
        raise NotImplementedError(
            "Iteration over this class is unlikely to be threadsafe."
        )

    def clear(self):
        with self._lock:
            # Copy pointers to all values, then wipe the mapping
            # under Python 2, this copies the list of values twice :-|
            values = list(self._container.values())
            self._container.clear()

        if self.dispose_func:
            for value in values:
                self.dispose_func(value)

    def keys(self):
        with self._lock:
            return self._container.keys()


def generator_range(generator, count):
    """
    * ---------------{Function}---------------
    * Iterates over the given generator and yields `count` values from it.
    * ----------------{Returns}---------------
    * -> generator   ::generator |A generator that yields `count` values from the original generator.
    * ----------------{Params}----------------
    * : generator    ::generator |The original generator to iterate over.
    * : count        ::int       |The number of values to yield from the generator.
    * ----------------{Usage}-----------------
    * >>> def my_generator():
    * ...     for i in range(10):
    * ...         yield i
    * ...
    * >>> for value in generator_range(my_generator(), 3):
    * ...     print(value)
    * 0
    * 1
    * 2
    """
    return [generator.__next__() for i in range(count)]
