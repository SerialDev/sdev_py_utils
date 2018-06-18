
import sys, inspect
import os
import types
import ast
import datetime
import threading, traceback
import numpy as np
import logging
import doctest
import textwrap
import functools
import shutil
import requests
from functools import reduce
from itertools import zip_longest
# import StringIO

#pip install --global-option build_ext --global-option --compiler=mingw32 falconn



# Python script to run a command line

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


sentinel = object()
def rgetattr(obj, attr, default=sentinel):
    if default is sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return functools.reduce(_getattr, [obj]+attr.split('.'))


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


def from_epoch(timestamp):
    return datetime.datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')


def epoch_add_second(timestamp, second):
    return int(timestamp + (second * 1000))


def inspect_arguments():
        """Returns tuple containing dictionary of calling function's
           named arguments and a list of calling function's unnamed
           positional arguments.
        """
        from inspect import getargvalues, stack
        posname, kwname, args = getargvalues(stack()[1][0])[-3:]
        posargs = args.pop(posname, [])
        args.update(args.pop(kwname, []))
        return args, posargs

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def function_params(f):
    return inspect.signature(f)

def top_level_functions(body):
    return (f for f in body if isinstance(f, ast.FunctionDef))

def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)

def typePrint(object):
    print(str(object) + " - (" + str(type(object)) + ")")


def flatten_list(sequence):
    return sum(sequence, [])


def sumprod(x,y):
    return reduce(lambda a,b:a + b,map(lambda a,b:a * b,x,y))


def seek_next_line(f):
    for c in iter(lambda: f.read(1), '\n'):
        pass


#import signal
#signal.signal(signal.SIGQUIT, dumpstacks)

def quick_inspect(package):
    package_dict = {}
    for callable in package.__dict__.values():
        try:
            package_dict['{}'.format(callable.__name__)] =  '{}'.format(inspect.signature(callable))
            #print('{} : {}'.format(callable.__name__, inspect.signature(callable)))
        except Exception as e:
            #print(TypeError)
            pass
    return package_dict

def try_except(success, failure, *exceptions):
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure


#def list_dir():

# import subprocess
# os.system('ls', getoutput)
# subprocess.getoutput('dir')


# 'isinstance': <function isinstance>,
#   'issubclass': <function issubclass>,
#   ismodule(), isclass(), ismethod(), isfunction(), isgeneratorfunction()
#   isgenerator(), istraceback(), isframe(), iscode(), isbuiltin(),\n        isroutine() - check object types\n    getmembers() - get members of an object that satisfy a given condition\n\n    getfile(), getsourcefile(), getsource() - find an object's source code\n
#   getdoc(), getcomments() - get documentation on an object\n    getmodule() - determine the module





##==========================={Object Introspection related}=====================

import gc
def objects_by_id(id_):
    """
    * Function: Get PyObject from memory address
    * -----------{returns}------------
    * PyObject at address [0x0000] using garbage collection module . . .
    """
    for obj in gc.get_objects():
        if id(obj) == id_:
            return obj
    raise Exception("No found")

def get_class( kls ):
    """
    * Function: Implement reflection in python
    * -----------{returns}------------
    * returns a class based on a string . . .
    """

    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def dynamic_slotted_struct(class_name, values):
    stripped = str(values).strip('[]').replace("'", '')
    template = textwrap.dedent(
"""
class {class_name}(object):
    __slots__ = {slots}

    def __init__(self, {init}):
        self.{selfs} = {vars}
""").format(class_name=class_name,
            slots=values,
            init=stripped,
            selfs='self.'.join(stripped.split()),
            vars = stripped  )
    #exec(template)
    return template

def is_mod_function(mod, func):
    """
    """
    return inspect.isfunction(func) and inspect.getmodule(func) == mod

def list_functions(mod):
    """
    print 'functions in current module:\n', list_functions(sys.modules[__name__])
    print 'functions in inspect module:\n', list_functions(inspect)
    """
    return [func.__name__ for func in mod.__dict__.values()
            if is_mod_function(mod, func)]

def list_functions_types(mod):
    """
    """
    func_list = []

    func_list.append([mod.__dict__.get(a) for a in dir(mod)
           if isinstance(mod.__dict__.get(a), types.FunctionType)])
    return func_list

def check_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__




##==========================={String manipulation}==============================


import re
def convert(name):
    """
    * Function: convert to snake_case
    * -----------{returns}------------
    * snake_case . . .
    * -----------{doctest}------------
    >>> convert('getHTTPResponseCode')
    'get_http_response_code'
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def to_camel_case(snake_str):
    components = snake_str.split('_')
    # We capitalize the first letter of each component except the first one
    # with the 'title' method and join them together.
    return components[0] + "".join(x.title() for x in components[1:])

def to_pascal_case(column):
   first, *rest = column.split('_')
   return first.capitalize() + ''.join(word.capitalize() for word in rest)

# ------------{String utils}------------#

def remove_xoo(name):
    return name.rstrip(' \t\r\n\0')


def predict_encoding(file_path, n_lines=20):
    '''Predict a file's encoding using chardet'''
    import chardet

    # Open the file as binary data
    with open(file_path, 'rb') as f:
        # Join binary lines for specified number of lines
        rawdata = b''.join([f.readline() for _ in range(n_lines)])

    return chardet.detect(rawdata)


# ==========================={Map and Apply Utilities}==========================#


# ==============================={Iterator utils}===============================#

# -----{template}----#
# from functools import wraps

# def decorator(argument):
#     def real_decorator(function):
#         @wraps(function)
#         def wrapper(*args, **kwargs):
#             funny_stuff()
#             something_with_argument(argument)
#             retval = function(*args, **kwargs)
#             more_funny_stuff()
#             return retval
#         return wrapper
#     return real_decorator

# Count Iterator
from collections import deque
def count(iterable):
    if hasattr(iterable, '__len__'):
        return len(iterable)

    d = deque(enumerate(iterable, 1), maxlen=1)
    return d[0][0] if d else 0


def limit_iteration(iterator, limit):
    for i in range(limit):
        try:
            yield(next(iterator))
        except StopIteration:
            raise StopIteration


from functools import wraps

def deco_limit_iteration(argument):
    def real_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            val = function(*args, **kwargs)
            gen = limit_iteration(val,  argument)
            return gen
        return wrapper
    return real_decorator


def get_original_fn(fn):
    """Gets the very original function of a decorated one."""

    fn_type = type(fn)
    if fn_type is classmethod or fn_type is staticmethod:
        return get_original_fn(fn.__func__)
    if hasattr(fn, 'original_fn'):
        return fn.original_fn
    if hasattr(fn, 'fn'):
        fn.original_fn = get_original_fn(fn.fn)
        return fn.original_fn
    return fn


# ============================{Assertion decorators}============================#

def parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer

import itertools as it
# TODO fix Named arguments
@parametrized
def types(f, *types):
    def rep(*args):
        for a, t, n in zip(args, types, it.count()):
            if type(a) is not t:
                raise TypeError('Value %d has not type %s. %s instead' %
                    (n, t, type(a))
                )
        return f(*args)
    return rep

##==========================={Parallelism}======================================

from multiprocessing import Process
def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()


##==========================={Link directly to system math DLL}=================

from sys import platform as _platform
import ctypes as _ctypes

if _platform == "linux" or _platform == "linux2":
    _libm = _ctypes.cdll.LoadLibrary('libm.so.6')
    _funcname = 'nextafter'
elif _platform == "darwin":
    _libm = _ctypes.cdll.LoadLibrary('libSystem.dylib')
    _funcname = 'nextafter'
elif _platform == "win32":
    _libm = _ctypes.cdll.LoadLibrary('msvcrt.dll')
    _funcname = '_nextafter'
else:
    # these are the ones I have access to...
    # fill in library and function name for your system math dll
    print( "Platform", repr(_platform), "is not supported" )
    sys.exit(0)

_nextafter = getattr(_libm, _funcname)
_nextafter.restype = _ctypes.c_double
_nextafter.argtypes = [_ctypes.c_double, _ctypes.c_double]

def next_after(x, y):
    """Returns the next floating-point number after x in the direction of y."""
    # This implementation comes from here:
    # http://stackoverflow.com/a/6163157/1256988
    return _nextafter(x, y)
#assert nextafter(0, 1) - nextafter(0, 1) == 0
#assert 0.0 + nextafter(0, 1) > 0.0

_pow = getattr(_libm, 'pow')
_pow.restype = _ctypes.c_double
_pow.argtypes = [_ctypes.c_double, _ctypes.c_double]
def pow_sys(x, y):
    # Not faster it seems
    return _pow(x, y)

_sqrt = getattr(_libm, 'sqrt')
_sqrt.restype = _ctypes.c_double
_sqrt.argtypes = [_ctypes.c_double]
def sqrt_sys(x):
    # Not faster it seems
    return _sqrt(x)

_exp = getattr(_libm, 'exp')
_exp.restype = _ctypes.c_double
_exp.argtypes = [_ctypes.c_double]
def exp_sys(x):
    return _exp(x)

_log = getattr(_libm, 'log')
_log.restype = _ctypes.c_double
_log.argtypes = [_ctypes.c_double]
def log_sys(x):
    return _log(x)

_rand = getattr(_libm, 'rand')
_rand.restype = _ctypes.c_double
def rand_sys():
    return _rand()

_stricmp = getattr(_libm, '_stricmp')
_stricmp.restype = _ctypes.c_int
_stricmp.argtypes = [_ctypes.c_char_p, _ctypes.c_char_p]
def stricmp_sys(str1, str2):
    """
    * Function: Performs a case-insensitive comparison of strings.

    * -----------{returns}------------
    * -0, 0, +0 . . .
    """
    str1 = _ctypes.cast(str1, _ctypes.c_char_p)
    str2 = _ctypes.cast(str2, _ctypes.c_char_p)
    return _stricmp(str1, str2)




##==========================={Ctypes-utils}=====================================

def get_c_array(python_list):
    try:
       return (_ctypes.c_int * len(python_list))(* python_list)
    except TypeError:
        return (_ctypes.c_float * len(python_list))(* python_list)

import ctypes

# Create pointer classes once up front for brevity/performance later
# Assumes ctypes.sizeof(ctypes.c_double) is 8 bytes; could add assert for this
PDOUBLE = ctypes.POINTER(ctypes.c_double)
PU64 = ctypes.POINTER(ctypes.c_uint64)

def float_to_bin(f):
    d = ctypes.c_double(f)       # Convert to true C double
    pd = PDOUBLE(d)              # Make pointer to it
    pu64 = ctypes.cast(pd, PU64) # Cast pointer to unsigned int type of same size
    return '{:b}'.format(pu64[0]) # Read value as unsigned int and convert to bin

def bin_to_float(b):
    u64 = ctypes.c_uint64(int(b, 2)) # Convert bin form to unsigned int
    pu64 = PU64(ul)                  # Make pointer to it
    pd = ctypes.cast(pu64, PDOUBLE)  # Cast pointer to double pointer
    return pd[0]                     # Read double value as Python float








# ================================{Binary utils}================================#


# def unpack_from(fmt, data, offset = 0):
#     (byte_order, fmt, args) = (fmt[0], fmt[1:], ()) if fmt and fmt[0] in ('@', '=', '<', '>', '!') else ('@', fmt, ())
#     fmt = filter(None, re.sub("p", "\tp\t",  fmt).split('\t'))
#     for sub_fmt in fmt:
#         if sub_fmt == 'p':
#             (str_len,) = struct.unpack_from('B', data, offset)
#             sub_fmt = str(str_len + 1) + 'p'
#             sub_size = str_len + 1
#         else:
#             sub_fmt = byte_order + sub_fmt
#             sub_size = struct.calcsize(sub_fmt)
#         args += struct.unpack_from(sub_fmt, data, offset)
#         offset += sub_size
#     return args

# def unpack_helper(fmt, data):
#     size = struct.calcsize(fmt)
#     return struct.unpack(fmt, data[:size]), data[size:]

# def pack(fmt, *args):
#     (byte_order, fmt, data) = (fmt[0], fmt[1:], '') if fmt and fmt[0] in ('@', '=', '<', '>', '!') else ('@', fmt, '')
#     fmt = filter(None, re.sub("p", "\tp\t",  fmt).split('\t'))
#     for sub_fmt in fmt:
#         if sub_fmt == 'p':
#             (sub_args, args) = ((args[0],), args[1:]) if len(args) > 1 else ((args[0],), [])
#             sub_fmt = str(len(sub_args[0]) + 1) + 'p'
#         else:
#             (sub_args, args) = (args[:len(sub_fmt)], args[len(sub_fmt):])
#             sub_fmt = byte_order + sub_fmt
#         data += struct.pack(sub_fmt, *sub_args)
#     return data





