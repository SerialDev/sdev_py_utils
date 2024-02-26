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
    """
    * ---------------{Function}---------------
    * Serializes the given data object using dill and returns it as a byte buffer.
    * ----------------{Returns}---------------
    * -> buf       ::BytesIO   |A byte buffer containing the serialized data object
    * ----------------{Params}----------------
    * : data      ::Any        |The data object to be serialized
    * ----------------{Usage}-----------------
    * >>> data = {'name': 'John', 'age': 30, 'city': 'New York'}
    * >>> buf = to_buffer(data)
    * >>> buf.getvalue()
    * b'\x80\x04\x95(\x00\x00\x00\x00\x00\x00\x00}\x94(\x8c\x04name\x94\x8c\x04John\x94\x8c\x03age\x94K\x1e\x8c\x04city\x94\x8c\x08New York\x94u.'
    """
    import io
    import dill as pickle  # Dill allows serializing lambdas

    buf = io.BytesIO()
    buf.seek(0)
    buf.write(pickle.dumps(data))
    return buf


def from_buffer(buf):
    """
    * ---------------{Function}---------------
    * Deserializes a buffer and returns the deserialized object.
    * ----------------{Returns}---------------
    * -> data    ::Any        |The deserialized object
    * ----------------{Params}----------------
    * : buf     ::io.BytesIO |The buffer containing the serialized object
    * ----------------{Usage}-----------------
    * >>> with open('data.pkl', 'rb') as f:
    * ...     buf = io.BytesIO(f.read())
    * >>> data = from_buffer(buf)
    * >>> print(data)
    * {'name': 'John', 'age': 30, 'city': 'New York'}
    """
    import dill as pickle  # Dill allows serializing lambdas

    buf.seek(0)
    data = pickle.loads(buf.read())
    return data


def salty_hash(content, salt="deadbeeffeebdaed"):
    """
    * ---------------{Function}---------------
    * Hashes a given content with SHA512 and a given salt
    * ----------------{Returns}---------------
    * -> hashed  ::str        |The hashed content in hexadecimal format
    * ----------------{Params}----------------
    * : content  ::str        |The content to be hashed
    * : salt     ::str        |A salt to be used in the hashing process (default is 'deadbeeffeebdaed')
    * ----------------{Usage}-----------------
    * >>> salty_hash("password")
    * '2b3e69c61f18e20aa8c83d86b46c273da9abbd7d02c20a11e7c5b37cf45a7f0a8ed03d496f3a55df9e7d5488d45d1927a4ce6b4dd6ad4ed6d0a6e0b2979ac110'
    * >>> salty_hash("password", "1a2b3c4d")
    * 'd6a5c6f9df30a5c5d5d14ec29c510b27e7a36a8a28f64cd3a3a9f7e10d8c3109637a3c181ef40a2f86478c8d42629d97a25c9bb0ec0d87c85a7f0a67c167b7e8'
    """
    import hashlib

    hashed = hashlib.sha512(content.encode("utf-8") + salt.encode("utf-8")).hexdigest()
    return hashed
