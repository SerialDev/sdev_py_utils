"""Python core profiling & debugging utilitites"""

import threading
import traceback
import sys
from sys import getsizeof, stderr
from itertools import chain
from collections import deque

try:
    from reprlib import repr
except ImportError:
    pass

from decorator import decorator
from datetime import datetime
import traceback
import numpy as np
import time


def lsos(all_obj=globals(), n=10):
    # Usage lsos(globals())

    import sys

    object_name = list(all_obj)
    object_size = [
        round(sys.getsizeof(all_obj[x]) / 1024.0 / 1024.0, 4) for x in object_name
    ]
    object_id = [id(all_obj[x]) for x in object_name]

    d = [(a, b, c) for a, b, c in zip(object_name, object_size, object_id)]
    d.sort(key=lambda x: (x[1], x[2]), reverse=True)
    dprint = d[0 : min(len(d), n)]

    # print formating
    name_width_max = max([len(x[0]) for x in dprint])
    print(
        ("{:<" + str(name_width_max + 2) + "}{:11}{}").format("name", "size_Mb", "id")
    )
    fmt = "{{:<{}}}".format(name_width_max + 2) + "  " + "{: 5.4f}" + "  " + "{:d}"
    for line in dprint:
        print(fmt.format(*line))

    return d


def total_size(o, handlers={}, verbose=False):
    """
    Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
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


def dumpstacks(signal, frame):
    id2name = dict([(th.ident, th.name) for th in threading.enumerate()])
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append("\n# Thread: %s(%d)" % (id2name.get(threadId, ""), threadId))
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append('File: "%s", line %d, in %s' % (filename, lineno, name))
            if line:
                code.append("  %s" % (line.strip()))
    print("\n".join(code))


class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print("elapsed time: %f ms" % self.msecs)


@decorator
def timing_function(some_function, *args, **kwargs):

    """
    Outputs the time a function takes
    to execute.
    """

    def wrapper(*args, **kwargs):
        t1 = time.time()
        some_function(*args, **kwargs)
        t2 = time.time()
        return "Time it took to run the function: " + str((t2 - t1)) + "\n"

    return wrapper


@decorator
class countcalls(object):
    "Decorator that keeps track of the number of times a function is called."

    __instances = {}

    def __init__(self, f):
        self.__f = f
        self.__numCalls = 0
        countcalls.__instances[f] = self

    def __call__(self, *args, **kwargs):
        self.__numCalls += 1
        return self.__f(*args, **kwargs)

    @staticmethod
    def count(f):
        "Return the number of times the function f was called."
        return countcalls.__instances[f].__numCalls

    @staticmethod
    def counts():
        "Return a dict of {function: # of calls} for all registered functions."
        return dict([(f, countcalls.count(f)) for f in countcalls.__instances])


def logged(time_format, name_prefix=""):
    def decorator(func):
        if hasattr(func, "_logged_decorator") and func._logged_decorator:
            return func

        @wraps(func)
        def decorated_func(*args, **kwargs):
            start_time = time.time()
            print(
                "- Running '%s' on %s "
                % (name_prefix + func.__name__, time.strftime(time_format))
            )
            result = func(*args, **kwargs)
            end_time = time.time()
            print(
                "- Finished '%s', execution time = %0.3fs "
                % (name_prefix + func.__name__, end_time - start_time)
            )

            return result

        decorated_func._logged_decorator = True
        return decorated_func

    return decorator


def log_method_calls(time_format):
    # @log_method_calls("%b %d %Y - %H:%M:%S")
    def decorator(cls):
        for o in dir(cls):
            if o.startswith("__"):
                continue
            a = getattr(cls, o)
            if hasattr(a, "__call__"):
                decorated_a = logged(time_format, cls.__name__ + ".")(a)
                setattr(cls, o, decorated_a)
        return cls

    return decorator


def dump_closure(f):
    if hasattr(f, "__closure__") and f.__closure__ is not None:
        print("- Dumping function closure for %s:" % f.__name__)
        for i, c in enumerate(f.__closure__):
            print("-- cell %d  = %s" % (i, c.cell_contents))
    else:
        print(" - %s has no closure!" % f.__name__)


# import sys

# WHAT_TO_DEBUG = set(['io', 'core'])  # change to what you need

# class debug:
#     '''Decorator which helps to control what aspects of a program to debug
#     on per-function basis. Aspects are provided as list of arguments.
#     It DOESN'T slowdown functions which aren't supposed to be debugged.
#     '''
#     def __init__(self, aspects=None):
#         self.aspects = set(aspects)

#     def __call__(self, f):
#         if self.aspects & WHAT_TO_DEBUG:
#             def newf(*args, **kwds):
#                 print >> sys.stderr, f.func_name, args, kwds
#                 f_result = f(*args, **kwds)
#                 print >> sys.stderr, f.func_name, "returned", f_result
#                 return f_result
#             newf.__doc__ = f.__doc__
#             return newf
#         else:
#             return f

# @debug(['io'])
# def prn(x):
#     print (x)

# @debug(['core'])
# def mult(x, y):
#     return x * y


class countcalls_deco(object):
    "Decorator that keeps track of the number of times a function is called."

    __instances = {}

    def __init__(self, f):
        self.__f = f
        self.__numcalls = 0
        countcalls.__instances[f] = self

    def __call__(self, *args, **kwargs):
        self.__numcalls += 1
        return self.__f(*args, **kwargs)

    def count(self):
        "Return the number of times the function f was called."
        return countcalls.__instances[self.__f].__numcalls

    @staticmethod
    def counts():
        "Return a dict of {function: # of calls} for all registered functions."
        return dict(
            [
                (f.__name__, countcalls.__instances[f].__numcalls)
                for f in countcalls.__instances
            ]
        )


def dump_args(func):
    "This decorator dumps out the arguments passed to a function before calling it"
    argnames = func.func_code.co_varnames[: func.func_code.co_argcount]
    fname = func.func_name

    def echo_func(*args, **kwargs):
        print(
            fname,
            ":",
            ", ".join(
                "%s=%r" % entry for entry in zip(argnames, args) + kwargs.items()
            ),
        )
        return func(*args, **kwargs)

    return echo_func


import sys
import os
import linecache


def trace(f):
    def globaltrace(frame, why, arg):
        if why == "call":
            return localtrace
        return None

    def localtrace(frame, why, arg):
        if why == "line":
            # record the file name and line number of every trace
            filename = frame.f_code.co_filename
            lineno = frame.f_lineno

            bname = os.path.basename(filename)
            print(
                "{}({}): {}".format(bname, lineno, linecache.getline(filename, lineno))
            )
        return localtrace

    def _f(*args, **kwds):
        sys.settrace(globaltrace)
        result = f(*args, **kwds)
        sys.settrace(None)
        return result

    return _f


# FIX THIS
def log_err(err_str):
    def Log(log_string):
        log_entry = "[%s] %s" % (
            datetime.now().strftime("%Y-%m-%d %H:%M.%S"),
            log_string,
        )
        print(log_entry)

    def log_error_trace(errorstring, exception):
        Log("{}: {}".format(errorstring, exception))
        Log("::>>{}".format(traceback.print_exc()))

    def real_decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                log_error_trace("{} :/n".format(err_str), e)

        return wrapper

    return real_decorator


@decorator
def print_args(function):
    def wrapper(*args, **kwargs):
        print("Arguments:", args, kwargs)
        return function(*args, **kwargs)

    return wrapper


import functools, logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class log_with(object):
    """Logging decorator that allows you to log with a
specific logger.
"""

    # Customize these messages
    ENTRY_MESSAGE = "Entering {}"
    EXIT_MESSAGE = "Exiting {}"

    def __init__(self, logger=None):
        self.logger = logger

    def __call__(self, func):
        """Returns a wrapper that wraps func.
The wrapper will log the entry and exit points of the function
with logging.INFO level.
"""
        # set logger if it was not set earlier
        if not self.logger:
            logging.basicConfig()
            self.logger = logging.getLogger(func.__module__)

        @functools.wraps(func)
        def wrapper(*args, **kwds):
            self.logger.info(
                self.ENTRY_MESSAGE.format(func.__name__)
            )  # logging level .info(). Set to .debug() if you want to
            f_result = func(*args, **kwds)
            self.logger.info(
                self.EXIT_MESSAGE.format(func.__name__)
            )  # logging level .info(). Set to .debug() if you want to
            return f_result

        return wrapper


# By modifying the decorator, we can keep a reference to the original function


def time_long_list(func):
    import random

    def helper(*args, **kwargs):
        for item in [10, 100, 1000, 10000]:
            list = [random.randint(1, 10) for element in range(item)]
            with Timer() as clock:
                func(list, func)
                print(clock.interval / item)
        return func

    helper.original = func
    return helper


# Then modifying your recursive function so that it always calls the original version of itself, not the modified version


@time_long_list
def minimum(lst, undecorated_func=None):
    if len(lst) == 0:
        return None
    elif len(lst) == 1:
        return lst[0]
    else:
        mid = len(lst) // 2
        min1 = minimum.original(lst[:mid])
        min2 = minimum.original(lst[mid:])
        if min1 <= min2:
            return min1
        else:
            return min2


class MyException(Exception):
    pass


def handleError(func):
    errors = []

    def wrapper(arg1):
        result = func(arg1)

        for err in findError(result):
            errors.append(err)

        print(errors)
        return result

    return wrapper


def findError(result):
    print(result)
    for k, v in result.iteritems():
        error_nr = v % 2
        if error_nr == 0:
            pass
        elif error_nr > 0:
            yield MyException


@handleError
def numGen(input):
    from random import randint

    result = {}
    for i in range(9):
        j = randint(0, 4)
        result[i] = input + j
    return result


class debug_context:
    """ Debug context to trace any function calls inside the context """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print("Entering Debug Decorated func")
        # Set the trace function to the trace_calls function
        # So all events are now traced
        sys.settrace(self.trace_calls)

    def __exit__(self, *args, **kwargs):
        # Stop tracing all events
        sys.settrace = None

    def trace_calls(self, frame, event, arg):
        # We want to only trace our call to the decorated function
        if event != "call":
            return
        elif frame.f_code.co_name != self.name:
            return
        # return the trace function to use when you go into that
        # function call
        return self.trace_lines

    def trace_lines(self, frame, event, arg):
        # If you want to print local variables each line
        # keep the check for the event 'line'
        # If you want to print local variables only on return
        # check only for the 'return' event
        if event not in ["line", "return"]:
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        filename = co.co_filename
        local_vars = frame.f_locals
        print("  {0} {1} {2} locals: {3}".format(func_name, event, line_no, local_vars))


def debug_decorator(func):
    """ Debug decorator to call the function within the debug context """

    def decorated_func(*args, **kwargs):
        with debug_context(func.__name__):
            return_value = func(*args, **kwargs)
        return return_value

    return decorated_func


# @debug_decorator
# def testing() :
#     a = 10
#     b = 20
#     c = a + b
#
# testing()
# ###########################################################
# #output:
# #   Entering Debug Decorated func
# #     testing line 44 locals: {}
# #     testing line 45 locals: {'a': 10}
# #     testing line 46 locals: {'a': 10, 'b': 20}
# #     testing return 46 locals: {'a': 10, 'b': 20, 'c': 30}
# ###########################################################


from functools import wraps


class StateMachineWrongState(Exception):
    def __init__(self, shouldbe, current):
        self.shouldbe = shouldbe
        self.current = current
        super().__init__((shouldbe, current))


def statemachine(shouldbe, willbe):
    def decorator(f):
        @wraps(f)
        def wrapper(self, *args, **kw):
            if self.state != shouldbe:
                raise StateMachineWrongState(shouldbe, self.state)
            try:
                return f(self, *args, **kw)
            finally:
                self.state = willbe

        return wrapper

    return decorator


#
# >>> cm = CoffeeMachine()
# >>> cm.state
# <CoffeeState.Initial: 0>
# >>> cm.ground_beans()
# ground_beans
# >>> cm.state
# <CoffeeState.Grounding: 1>
# >>> cm.ground_beans()
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
#   File "<stdin>", line 6, in wrapper
# __main__.StateMachineWrongState: (<CoffeeState.Initial: 0>, <CoffeeState.Grounding: 1>)
# >>> cm.heat_water()
# heat_water
# >>> cm.pump_water()
# pump_water
# >>> cm.state
# <CoffeeState.Pumping: 3>


class profile(object):
    # class variable used as a stack of subs list
    stack = [[]]

    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kw):
        func = dict(fname=self.f.__name__)

        # append the current function in the latest pushed subs list
        profile.stack[-1].append(func)

        # push a new subs list in the stack
        profile.stack.append([])

        # execution time of the actual call
        t0 = time.time()
        out = self.f(*args, **kw)
        func["etime"] = time.time() - t0

        # pull the subs list from the stack
        func["subs"] = profile.stack.pop()

        return out

    @classmethod
    def show(cls):
        import json  # useful to prettify the ouput

        for func in cls.stack[0]:
            print(json.dumps(func, sort_keys=True, indent=4))


#    >>> import klepto
#    >>> from klepto import lru_cache as memoize
#    >>> from klepto.keymaps import hashmap
#    >>> hasher = hashmap(algorithm='md5')
#    >>> @memoize(keymap=hasher)
#    ... def squared(x):
#    ...   print("called")
#    ...   return x**2
#    ...
#    >>> squared(1)
#    called
#    1
#    >>> squared(2)
#    called
#    4
#    >>> squared(3)
#    called
#    9
#    >>> squared(2)
#    4
#    >>>
#    >>> cache = squared.__cache__()
#    >>> # delete the 'key' for x=2
#    >>> cache.pop(squared.key(2))
#    4
#    >>> squared(2)
#    called
#    4
#
