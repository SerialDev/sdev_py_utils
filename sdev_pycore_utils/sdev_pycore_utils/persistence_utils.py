#


def encode_b64(data):
    import base64

    return base64.b64encode(data)


def decode_b64(data):
    import base64

    return base64.b64decode(data)


def compress_bin(data):
    import zlib

    return zlib.compress(data)


def decompress_bin(data):
    import zlib

    return zlib.decompress(data)


def serialize_msgpack(data):
    import msgpack

    return msgpack.packb(data, use_bin_type=True)


def deserialize_msgpack(data):
    import msgpack

    return msgpack.unpackb(data, raw=False)


def network_compress_decode(data):
    import zlib
    import base64

    result = zlib.decompress(base64.b64decode(data.encode()))
    return result


def network_compress_encode(data):
    import zlib
    import base64

    result = base64.b64encode(zlib.compress(data.encode())).decode()
    return result


def network_compress_pickle(data):
    import zlib
    import base64
    import lzma
    import dill as pickle

    result = base64.b64encode(zlib.compress(pickle.dumps(data))).decode()
    return result


def network_decompress_pickle(data):
    import zlib
    import base64
    import lzma
    import dill as pickle

    result = pickle.loads(zlib.decompress(base64.b64decode(data)))
    return result


def save_pickle_to_b64(data):
    try:
        return b64encode_buffer(pickle_to_buffer(data)).decode()
    except Exception:
        return b64encode_buffer(pickle_to_buffer(data))


def pickle_to_buffer(data):
    import io
    import dill as pickle

    buffer = io.BytesIO()
    buffer.write(pickle.dumps(data))
    buffer.seek(0)
    return buffer


def b64encode_buffer(buffer):
    import base64

    return base64.b64encode(buffer.read())


def md5(string):
    import hashlib

    return hashlib.md5(string.encode("utf-8")).hexdigest()


def to_buffer(data):
    import io
    import dill as pickle  # Dill allows serializing lambdas

    buf = io.BytesIO()
    buf.seek(0)
    buf.write(pickle.dumps(data))
    return buf


def from_buffer(buf):
    import dill as pickle  # Dill allows serializing lambdas

    buf.seek(0)
    data = pickle.loads(buf.read())
    return data


def cast_bytesio_encoding(data):
    from base64 import b64encode

    data.seek(0)
    return b64encode(data.read())


def b64decode_data(data):
    import base64

    return base64.b64decode(data)


def b64encode_data(data):
    import base64

    return base64.b64encode(data)


def load_pickle_from_b64(data):
    try:
        return pickle.loads(b64decode_data(data))
    except Exception:
        return pickle.loads(b64decode_data(data.encode()))


def cast_bytesio_encoding(data):
    from base64 import b64encode

    data.seek(0)
    return b64encode(data.read())


def cast_encoding_bytesio(data):
    from base64 import b64decode
    from io import BytesIO

    buf = BytesIO()
    buf.write(b64decode(data))
    buf.seek(0)
    return buf


def total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    from sys import getsizeof, stderr
    from collections import deque
    from itertools import chain

    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                # s += sum(map(sizeof, handler(o)))
                break
        else:
            if not hasattr(o.__class__, "__slots__"):
                if hasattr(o, "__dict__"):
                    s += sizeof(
                        o.__dict__
                    )  # no __slots__ *usually* means a __dict__, but some special builtin classes (such as `type(None)`) have neither
                # else, `o` has no attributes at all, so sys.getsizeof() actually returned the correct value
            else:
                s += sum(
                    sizeof(getattr(o, x))
                    for x in o.__class__.__slots__
                    if hasattr(o, x)
                )
        return s

    return sizeof(o)


from datetime import datetime
import pandas as pd
import numpy as np


def to_buffer(data):
    import io
    import dill as pickle  # Dill allows serializing lambdas

    buf = io.BytesIO()
    buf.seek(0)
    buf.write(pickle.dumps(data))
    return buf


def from_buffer(buf):
    import dill as pickle  # Dill allows serializing lambdas

    buf.seek(0)
    data = pickle.loads(buf.read())
    return data


def salty_hash(content, salt="deadbeeffeebdaed"):
    import hashlib

    hashed = hashlib.sha512(content.encode("utf-8") + salt.encode("utf-8")).hexdigest()
    return hashed
