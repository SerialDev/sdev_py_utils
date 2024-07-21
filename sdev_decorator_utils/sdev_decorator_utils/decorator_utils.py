from decorator import decorator
from datetime import datetime
import traceback
import numpy as np
import time


import functools


def retry_decorator(max_retries: int = 5, backoff_factor: int = 2):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retry_count = 0
            while retry_count < max_retries:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    wait_time = backoff_factor**retry_count
                    print(
                        f"Request failed with error {e}. Retrying in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)
            print(f"Max retries exceeded. Giving up.")
            raise

        return wrapper

    return decorator


def lazy_property(function):
    attribute = "_" + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
            return getattr(self, attribute)
        return wrapper


import time


@decorator
class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        try:
            return self.cache[args]
        except KeyError:
            self.cache[args] = value = self.func(*args)
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args)

    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__


def _dynamic_programming(f, *args, **kwargs):
    if f.data is None:
        f.data = args[0]
    if not np.array_equal(f.data, args[0]):
        f.cache = {}
        f.data = args[0]

    try:
        f.cache[args[1:3]]
    except KeyError:
        f.cache[args[1:3]] = f(*args, **kwargs)
    return f.cache[args[1:3]]


def dynamic_fn(f):
    f.cache = {}
    f.data = None

    return decorator(_dynamic_programming, f)


import itertools


def retry(
    delays=(0, 1, 5, 30, 180, 600, 3600), exception=Exception, report=lambda *args: None
):
    def wrapper(function):
        def wrapped(*args, **kwargs):
            problems = []
            for delay in itertools.chain(delays, [None]):
                try:
                    return function(*args, **kwargs)
                except exception as problem:
                    problems.append(problem)
                    if delay is None:
                        report("retryable failed definitely:", problems)
                        raise
                    else:
                        report(
                            "retryable failed:", problem, "-- delaying for %ds" % delay
                        )
                        time.sleep(delay)

        return wrapped

    return wrapper


# Todo FIX@ Mon,  5 Jun 2017, 11:41
def immutable(mutableclass):
    """Decorator for making a slot-based class immutable"""

    if not isinstance(type(mutableclass), type):
        raise (TypeError("@immutable: must be applied to a new-style class"))
    if not hasattr(mutableclass, "__slots__"):
        raise (TypeError("@immutable: class must have __slots__"))

    class immutableclass(mutableclass):
        __slots__ = ()  # No __dict__, please

        def __new__(cls, *args, **kw):
            new = mutableclass(*args, **kw)  # __init__ gets called while still mutable
            new.__class__ = immutableclass  # locked for writing now
            return new

        def __init__(self, *args, **kw):  # Prevent re-init after __new__
            pass

    # Copy class identity:
    immutableclass.__name__ = mutableclass.__name__
    immutableclass.__module__ = mutableclass.__module__

    # Make read-only:
    for name, member in mutableclass.__dict__.items():
        if hasattr(member, "__set__"):
            setattr(immutableclass, name, property(member.__get__))


import threading
from contextlib import contextmanager

_tls = threading.local()


@contextmanager
def _nested():
    _tls.level = getattr(_tls, "level", 0) + 1
    try:
        yield "   " * _tls.level
    finally:
        _tls.level -= 1


@contextmanager
def _recursion_lock(obj):
    if not hasattr(_tls, "history"):
        _tls.history = []  # can't use set(), not all objects are hashable
    if obj in _tls.history:
        yield True
        return
    _tls.history.append(obj)
    try:
        yield False
    finally:
        _tls.history.pop(-1)


def humanize(cls):
    def __repr__(self):
        if getattr(_tls, "level", 0) > 0:
            return str(self)
        else:
            attrs = ", ".join("%s = %r" % (k, v) for k, v in self.__dict__.items())
            return "%s(%s)" % (self.__class__.__name__, attrs)

    def __str__(self):
        with _recursion_lock(self) as locked:
            if locked:
                return "<...>"
            with _nested() as indent:
                attrs = []
                for k, v in self.__dict__.items():
                    if k.startswith("_"):
                        continue
                    if isinstance(v, (list, tuple)) and v:
                        attrs.append("%s%s = [" % (indent, k))
                        with _nested() as indent2:
                            for item in v:
                                attrs.append("%s%r," % (indent2, item))
                        attrs.append("%s]" % (indent,))
                    elif isinstance(v, dict) and v:
                        attrs.append("%s%s = {" % (indent, k))
                        with _nested() as indent2:
                            for k2, v2 in v.items():
                                attrs.append("%s%r: %r," % (indent2, k2, v2))
                        attrs.append("%s}" % (indent,))
                    else:
                        attrs.append("%s%s = %r" % (indent, k, v))
                if not attrs:
                    return "%s()" % (self.__class__.__name__,)
                else:
                    return "%s:\n%s" % (self.__class__.__name__, "\n".join(attrs))

    cls.__repr__ = __repr__
    cls.__str__ = __str__
    return cls


def typecheck(func):

    if not hasattr(func, "__annotations__"):
        return method

    import inspect

    argspec = inspect.getfullargspec(func)

    def check(t, T):
        if type(T) == type:
            return isinstance(t, T)  # types
        else:
            return T(t)  # predicates

    def wrapper(*args):

        if len(argspec.args) != len(args):
            raise (
                TypeError(
                    "%s() takes exactly %s positional argument (%s given)"
                    % (func.__name__, len(argspec.args), len(args))
                )
            )

        for argname, t in zip(argspec.args, args):
            if argname in func.__annotations__:
                T = func.__annotations__[argname]
                if not check(t, T):
                    raise TypeError(
                        "%s( %s:%s ) but received %s=%s"
                        % (func.__name__, argname, T, argname, repr(t))
                    )

        r = func(*args)

        if "return" in func.__annotations__:
            T = func.__annotations__["return"]
            if not check(r, T):
                raise TypeError(
                    "%s() -> %s but returned %s" % (func.__name__, T, repr(r))
                )

        return r

    return wrapper


##==========================={Multithreading}===================================


def run_async(func):
    """
    run_async(func)
            function decorator, intended to make "func" run in a separate
            thread (asynchronously).
            Returns the created Thread object

            E.g.:
            @run_async
            def task1():
                    do_something

            @run_async
            def task2():
                    do_something_too

            t1 = task1()
            t2 = task2()
            ...
            t1.join()
            t2.join()
    """
    from threading import Thread
    from functools import wraps

    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Thread(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl

    return async_func


##==========================={CTypes decorators}================================


import ctypes


# TODO FIX
class C_struct:
    """Decorator to convert the given class into a C struct."""

    # contains a dict of all known translatable types
    types = ctypes.__dict__

    @classmethod
    def register_type(cls, typename, obj):
        """Adds the new class to the dict of understood types."""
        cls.types[typename] = obj

    def __call__(self, cls):
        """Converts the given class into a C struct.

        Usage:
                >>> @C_struct()
                ... class Account:
                ... 	first_name = "c_char_p"
                ...	last_name = "c_char_p"
                ... 	balance = "c_float"
                ...
                >>> a = Account()
                >>> a
                <cstruct.Account object at 0xb7c0ee84>

        A very important note: while it *is* possible to
        instantiate these classes as follows:

                >>> a = Account("Geremy", "Condra", 0.42)

        This is strongly discouraged, because there is at
        present no way to ensure what order the field names
        will be read in.
        """

        # build the field mapping (names -> types)
        fields = []
        for k, v in vars(cls).items():
            # don't wrap private variables
            if not k.startswith("_"):
                # if its a pointer
                if v.startswith("*"):
                    field_type = ctypes.POINTER(self.types[v[1:]])
                else:
                    field_type = self.types[v]
                new_field = (k, field_type)
                fields.append(new_field)

                # make our bases tuple
        bases = (ctypes.Structure,) + tuple((base for base in cls.__bases__))
        # finish up our wrapping dict
        class_attrs = {"_fields_": fields, "__doc__": cls.__doc__}

        # now create our class
        return type(cls.__name__, bases, class_attrs)


##==========================={Memoization/Cache}================================
import time
from functools import wraps


def cached(timeout, logged=False):
    """Decorator to cache the result of a function call.
    Cache expires after timeout seconds.
    """

    def decorator(func):
        if logged:
            print("-- Initializing cache for", func.__name__)
        cache = {}

        @wraps(func)
        def decorated_function(*args, **kwargs):
            if logged:
                print("-- Called function", func.__name__)
            key = (args, frozenset(kwargs.items()))
            result = None
            if key in cache:
                if logged:
                    print("-- Cache hit for", func.__name__, key)

                (cache_hit, expiry) = cache[key]
                if time.time() - expiry < timeout:
                    result = cache_hit
                elif logged:
                    print("-- Cache expired for", func.__name__, key)
            elif logged:
                print("-- Cache miss for", func.__name__, key)

            # No cache hit, or expired
            if result is None:
                result = func(*args, **kwargs)

            cache[key] = (result, time.time())
            return result

        return decorated_function

    return decorator


##==========================={Logging}==========================================

##==========================={Closures}=========================================


##==========================={More decorators}==================================


import warnings


def ignore_deprecation_warnings(func):
    """This is a decorator which can be used to ignore deprecation warnings
    occurring in a function."""

    def new_func(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            return func(*args, **kwargs)

    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


# from Queue import Queue
# import Queue
from threading import Thread


class asynchronous(object):
    def __init__(self, func):
        self.func = func

        def threaded(*args, **kwargs):
            self.queue.put(self.func(*args, **kwargs))

        self.threaded = threaded

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def start(self, *args, **kwargs):
        # self.queue = Queue()
        thread = Thread(target=self.threaded, args=args, kwargs=kwargs)
        thread.start()
        return asynchronous.Result(self.queue, thread)

    class NotYetDoneException(Exception):
        def __init__(self, message):
            self.message = message

    class Result(object):
        def __init__(self, queue, thread):
            self.queue = queue
            self.thread = thread

        def is_done(self):
            return not self.thread.is_alive()

        def get_result(self):
            if not self.is_done():
                raise asynchronous.NotYetDoneException(
                    "the call has not yet completed its task"
                )

            if not hasattr(self, "result"):
                self.result = self.queue.get()

            return self.result


# if __name__ == '__main__':
#     # sample usage
#     import time
#
#     @asynchronous
#     def long_process(num):
#         time.sleep(10)
#         return num * num
#
#     result = long_process.start(12)
#
#     for i in range(20):
#         print (i)
#         time.sleep(1)
#
#         if result.is_done():
#             print ("result {0}".format(result.get_result()))
#
#
#     result2 = long_process.start(13)
#
#     try:
#         print ("result2 {0}".format(result2.get_result()))
#
#     except asynchronous.NotYetDoneException as ex:
#         print (ex.message)


import threading, sys, functools, traceback


def lazy_thunkify(f):
    """Make a function immediately return a function of no args which, when called,
    waits for the result, which will start being processed in another thread."""

    @functools.wraps(f)
    def lazy_thunked(*args, **kwargs):
        wait_event = threading.Event()

        result = [None]
        exc = [False, None]

        def worker_func():
            try:
                func_result = f(*args, **kwargs)
                result[0] = func_result
            except Exception as e:
                exc[0] = True
                exc[1] = sys.exc_info()
                print(
                    "Lazy thunk has thrown an exception (will be raised on thunk()):\n%s"
                    % (traceback.format_exc())
                )
            finally:
                wait_event.set()

        def thunk():
            wait_event.wait()
            if exc[0]:
                raise Exception(exc[1][0], exc[1][1], exc[1][2])

            return result[0]

        threading.Thread(target=worker_func).start()

        return thunk

    return lazy_thunked


import pandas as pd
from functools import wraps


def seriescapable(func):
    """Decorator for turning the first argument from a pandas.Series to a pandas.DataFrame."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], pd.Series):
            return func(pd.DataFrame(args[0]), *args[1:], **kwargs)
        return func(*args, **kwargs)

    return wrapper


class ConvertArgumentTypes(object):
    """Converts function arguments to specified types."""

    def __init__(self, *args, **kw):
        self.args = args
        self.kw = kw

    def __call__(self, f):
        def func(*args, **kw):
            nargs = [x[0](x[1]) for x in zip(self.args, args)]
            invalidkw = [x for x in kw if x not in self.kw]
            if len(invalidkw) > 0:
                raise TypeError(
                    f.func_name
                    + "() got an unexpected keyword argument '%s'" % invalidkw[0]
                )
            kw = dict([(x, self.kw[x](kw[x])) for x in kw])
            v = f(*nargs, **kw)
            return v

        return func


#    #keyword arguments are handled normally.
#    @ConvertArgumentTypes(int, float, c=int, d=str)
#    def z(a,b,c=0,d=""):
#        return a + b, (c,d)
#
#    def add42(fn):
#        def wrap(i):
#            return fn(i) + 42
#        wrap.unwrapped = fn
#        return wrap
#
#    @add42
#    def mult3(i):
#        return i * 3
#
#    mult3(1) # 45
#    mult3.unwrapped(1) # 3


def cachedproperty(func):
    " Used on methods to convert them to methods that replace themselves\
        with their return value once they are called. "

    def cache(*args):
        self = args[0]  # Reference to the class who owns the method
        funcname = func.__name__
        ret_value = func(self)
        setattr(self, funcname, ret_value)  # Replace the function with its value
        return ret_value  # Return the result of the function

    return property(cache)


#    class Test:
#        @cachedproperty
#        def test(self):
#                print "Execute"
#                return "Return"
#
#    >>> test = Test()
#    >>> test.test
#    Execute
#    'Return'
#    >>> test.test
#    'Return'


##==========================={Debugging all calls}==============================
import sys


from threading import Semaphore, Timer
from functools import wraps


def ratelimit(limit, every):
    def limitdecorator(fn):
        semaphore = Semaphore(limit)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            semaphore.acquire()
            try:
                return fn(*args, **kwargs)
            finally:  # don't catch but ensure semaphore release
                timer = Timer(every, semaphore.release)
                timer.setDaemon(True)  # allows the timer to be canceled on exit
                timer.start()

        return wrapper

    return limitdecorator


from ast import parse, NodeTransformer


class Transformer(NodeTransformer):
    def __init__(self):
        self.src = ""
        self.indent = 0

    def translate(self, node):
        self.visit(node)
        return self.src

    def _indent(self, line):
        return "{}{line}".format(" " * self.indent, line=line)

    def render(self, body):
        self.indent += 2
        for stmt in body:
            self.visit(stmt)
        self.indent -= 2

    def visit_Num(self, node):
        self.src += "{}".format(node.n)

    def visit_Str(self, node):
        self.src += "{}".format(node.s)

    def visit_FunctionDef(self, defn):
        args = ",".join(name.arg for name in defn.args.args)
        js_defn = "var {} = function({}){{\n"
        self.src += self._indent(js_defn.format(defn.name, args))
        self.render(defn.body)
        self.src += self._indent("}\n")

    def visit_Eq(self, less):
        self.src += "=="

    def visit_Name(self, name):
        self.src += "{}".format(name.id)

    def visit_BinOp(self, binop):
        self.visit(binop.left)
        self.src += " "
        self.visit(binop.op)
        self.src += " "
        self.visit(binop.right)

    def visit_If(self, _if):
        self.src += self._indent("if (")
        self.visit(_if.test)
        self.src += ") {\n"
        self.render(_if.body)
        self.src += " " * self.indent + "}\n"

    def visit_Compare(self, comp):
        self.visit(comp.left)
        self.src += " "
        self.visit(comp.ops[0])
        self.src += " "
        self.visit(comp.comparators[0])

    def visit_Call(self, call):
        self.src += " "
        self.src += "{}(".format(call.func.id)
        self.visit(call.args[0])
        self.src += ")"

    def visit_Add(self, add):
        self.src += "+"

    def visit_Sub(self, add):
        self.src += "-"

    def visit_Return(self, ret):
        self.src += self._indent("return")
        if ret.value:
            self.src += " "
            self.visit(ret.value)
        self.src += ";\n"


def dec(f):
    source = getsource(f)
    _ast = parse(source)
    trans = Transformer()
    trans.indent = 0
    return trans.translate(_ast)


from inspect import getsource


def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


# Running the dec function outputs our python as javascript:

# print(dec(fibonacci))
# var fibonacci = function(n){
#   if (n == 0) {
#     return 0;
#   }
#   if (n == 1) {
#     return 1;
#   }
#   return  fibonacci(n - 1) +  fibonacci(n - 2);
# }
#


import functools
import time


def memoize(meth):
    @functools.wraps(meth)
    def wrapped(self, *args, **kwargs):

        # Prepare and get reference to cache
        attr = "_memo_{0}".format(meth.__name__)
        if not hasattr(self, attr):
            setattr(self, attr, {})
        cache = getattr(self, attr)

        # Actual caching
        key = args, tuple(sorted(kwargs))
        try:
            return cache[key]
        except KeyError:
            cache[key] = meth(self, *args, **kwargs)
            return cache[key]

    return wrapped


def network_call(user_id):
    print("Was called with: %s" % user_id)
    return 1


class NetworkEngine(object):
    @memoize
    def search(self, user_id):
        return network_call(user_id)


#
#    if __name__ == "__main__":
#        e = NetworkEngine()
#        for v in [1,1,2]:
#            e.search(v)
#        NetworkEngine().search(1)


from functools import wraps


def retry_if_exception(ex, max_retries):
    def outer(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            assert max_retries > 0
            x = max_retries
            while x:
                try:
                    return func(*args, **kwargs)
                except ex:
                    x -= 1

        return wrapper

    return outer


def load_or_make(filename):
    def decorator(func):
        def wraps(*args, **kwargs):
            try:
                with open(filename, "r") as f:
                    return json.load(input_handle)
            except Exception:
                data = func(*args, **kwargs)
                with open(filename, "w") as out:
                    json.dump(data, out)
                return data

        return wraps

    return decorator


#    @load_or_make(filename)
#    def your_method_with_arg(arg):
#        # do stuff
#        return data
#
#    @load_or_make(other_filename)
#    def your_method():
#        # do stuff
#        return data


##==========================={multithreaded deco}===============================
# import types
# from multiprocessing import Pool
#
# class concurrent(object):
#    functions = {}
#
#    @staticmethod
#    def custom(constructor = None, apply_async = None):
#        @staticmethod
#        def _custom_concurrent(*args, **kwargs):
#            conc = concurrent(*args, **kwargs)
#            if constructor is not None: conc.conc_constructor = constructor
#            if apply_async is not None: conc.apply_async = apply_async
#            return conc
#        return _custom_concurrent
#
#    def __init__(self, *args, **kwargs):
#        self.in_progress = False
#        self.conc_args = []
#        self.conc_kwargs = {}
#        if len(args) > 0 and isinstance(args[0], types.FunctionType):
#            self.setFunction(args[0])
#        else:
#            self.conc_args = args
#            self.conc_kwargs = kwargs
#        self.results = []
#        self.assigns = []
#        self.calls = []
#        self.arg_proxies = {}
#        self.conc_constructor = Pool
#        self.apply_async = lambda self, function, args: self.concurrency.apply_async(function, args)
#        self.concurrency = None
#
#    def __get__(self, *args):
#        raise NotImplementedError("Decorators from deco cannot be used on class methods")
#
#    def replaceWithProxies(self, args):
#        args_iter = args.items() if type(args) is dict else enumerate(args)
#        for i, arg in args_iter:
#            if type(arg) is dict or type(arg) is list:
#                if not id(arg) in self.arg_proxies:
#                    self.arg_proxies[id(arg)] = argProxy(id(arg), arg)
#                args[i] = self.arg_proxies[id(arg)]
#
#    def setFunction(self, f):
#        concurrent.functions[f.__name__] = f
#        self.f_name = f.__name__
#
#    def assign(self, target, *args, **kwargs):
#        self.assigns.append((target, self(*args, **kwargs)))
#
#    def call(self, target, *args, **kwargs):
#        self.calls.append((target, self(*args, **kwargs)))
#
#    def __call__(self, *args, **kwargs):
#        if len(args) > 0 and isinstance(args[0], types.FunctionType):
#            self.setFunction(args[0])
#            return self
#        self.in_progress = True
#        if self.concurrency is None:
#            self.concurrency = self.conc_constructor(*self.conc_args, **self.conc_kwargs)
#        args = list(args)
#        self.replaceWithProxies(args)
#        self.replaceWithProxies(kwargs)
#        result = ConcurrentResult(self.apply_async(self, concWrapper, [self.f_name, args, kwargs]))
#        self.results.append(result)
#        return result
#
#    def apply_operations(self, ops):
#        for arg_id, key, value in ops:
#            self.arg_proxies[arg_id].value.__setitem__(key, value)
#
#    def wait(self):
#        results = []
#        while self.results:
#            result, operations = self.results.pop().get()
#            self.apply_operations(operations)
#            results.append(result)
#        for assign in self.assigns:
#            assign[0][0][assign[0][1]] = assign[1].result()
#        self.assigns = []
#        for call in self.calls:
#            call[0](call[1].result())
#        self.calls = []
#        self.arg_proxies = {}
#        self.in_progress = False
#        return results


def run_async_process(func):
    """
    run_async_process(func)
            function decorator, intended to make "func" run in a separate
            thread (asynchronously).
            Returns the created Thread object

            E.g.:
            @run_async
            def task1():
                    do_something

            @run_async
            def task2():
                    do_something_too

            t1 = task1()
            t2 = task2()
            ...
            t1.join()
            t2.join()
    """
    from functools import wraps
    from multiprocessing import Process

    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Process(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl

    return async_func


def run_async_thread(func):
    """
    run_async(func)
            function decorator, intended to make "func" run in a separate
            thread (asynchronously).
            Returns the created Thread object

            E.g.:
            @run_async
            def task1():
                    do_something

            @run_async
            def task2():
                    do_something_too

            t1 = task1()
            t2 = task2()
            ...
            t1.join()
            t2.join()
    """
    from threading import Thread
    from functools import wraps

    @wraps(func)
    def async_func(*args, **kwargs):
        func_hl = Thread(target=func, args=args, kwargs=kwargs)
        func_hl.start()
        return func_hl

    return async_func


import sys


class TailRecurseException:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs


def tail_call_optimized(g):
    """
    This function decorates a function with tail call
    optimization. It does this by throwing an exception
    if it is it's own grandparent, and catching such
    exceptions to fake the tail call optimization.

    This function fails if the decorated
    function recurses in a non-tail context.
    """

    def func(*args, **kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back and f.f_back.f_back.f_code == f.f_code:
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs

    func.__doc__ = g.__doc__
    return func


class NoneSoFar:
    """
    This is a singleton to give you something to put somewhere that
    should never be a rightfull return value of anything.
    """

    def __str__(self):
        return "NoneSoFar"

    def __repr__(self):
        return "NoneSoFar"

    def __nonzero__(self):
        return 0


NoneSoFar = NoneSoFar()


def getitem(obj, key):
    """
    This is a helper function needed in promise objects to pass
    on __getitem__ calls. It just mimicks the getattr call, only
    it uses dictionary style access.
    """
    return obj[key]


def setitem(obj, key, value):
    """
    This is a helper function needed in promise objects to pass
    on __setitem__ calls. It just mimicks the setattr call, only
    it uses dictionary style access.
    """
    obj[key] = value


def delitem(obj, key):
    """
    This is a helper function needed in promise objects to pass
    on __delitem__ calls. It just mimicks the delattr call, only
    it uses dictionary style access.
    """
    del obj[key]


def getslice(obj, start, stop):
    """
    This is a helper function needed in promise objects to pass
    on __getslice__ calls. It just mimicks the getattr call, only
    it uses dictionary style access.
    """
    return obj[start:stop]


def setslice(obj, start, stop, value):
    """
    This is a helper function needed in promise objects to pass
    on __setslice__ calls. It just mimicks the setattr call, only
    it uses dictionary style access.
    """
    obj[start:stop] = value


def delslice(obj, start, stop):
    """
    This is a helper function needed in promise objects to pass
    on __delslice__ calls. It just mimicks the delattr call, only
    it uses dictionary style access.
    """
    del obj[start:stop]


def cmp(a, b):
    return (a > b) - (a < b)


def force(value):
    """
    This helper function forces evaluation of a promise. A promise
    for this function is something that has a __force__ method (much
    like an iterator in python is anything that has a __iter__
    method).
    """

    f = getattr(value, "__force__", None)
    if f:
        return f()
    else:
        return value


class PromiseMetaClass(type):
    """
    This meta class builds the behaviour of promise classes. It's mainly
    building standard methods with special behaviour to mimick several
    types in Python.

    The __magicmethods__ list defines what magic methods are created. Only
    those magic methods are defined that are not already defined by the
    class itself.

    __magicrmethods__ is much like __magicmethods__ only that it provides
    both the rmethod and the method so the proxy can decide what to use.

    The __magicfunctions__ list defines methods that should be mimicked by
    using some predefined function.

    The promise must define a __force__ method that will force evaluation
    of the promise.
    """

    __magicmethods__ = ["__abs__", "__pos__", "__invert__", "__neg__"]

    __magicrmethods__ = [
        ("__radd__", "__add__"),
        ("__rsub__", "__sub__"),
        ("__rdiv__", "__div__"),
        ("__rmul__", "__mul__"),
        ("__rand__", "__and__"),
        ("__ror__", "__or__"),
        ("__rxor__", "__xor__"),
        ("__rlshift__", "__lshift__"),
        ("__rrshift__", "__rshift__"),
        ("__rmod__", "__mod__"),
        ("__rdivmod__", "__divmod__"),
        ("__rtruediv__", "__truediv__"),
        ("__rfloordiv__", "__floordiv__"),
        ("__rpow__", "__pow__"),
    ]

    __magicfunctions__ = [
        ("__cmp__", cmp),
        ("__str__", str),
        # ('__unicode__', unicode), ('__complex__', complex),
        ("__int__", int),
        ("__float__", float),
        ("__oct__", oct),
        ("__hex__", hex),
        ("__hash__", hash),
        ("__len__", len),
        ("__iter__", iter),
        ("__delattr__", delattr),
        ("__setitem__", setitem),
        ("__delitem__", delitem),
        ("__setslice__", setslice),
        ("__delslice__", delslice),
        ("__getitem__", getitem),
        ("__getslice__", getslice),
        ("__nonzero__", bool),
    ]

    def __init__(klass, name, bases, attributes):
        for k in klass.__magicmethods__:
            if not attributes.has_key(k):
                setattr(klass, k, klass.__forcedmethodname__(k))
        for k, v in klass.__magicrmethods__:
            if not attributes.has_key(k):
                setattr(klass, k, klass.__forcedrmethodname__(k, v))
            if not attributes.has_key(v):
                setattr(klass, v, klass.__forcedrmethodname__(v, k))
        for k, v in klass.__magicfunctions__:
            if not attributes.has_key(k):
                setattr(klass, k, klass.__forcedmethodfunc__(v))
        super(PromiseMetaClass, klass).__init__(name, bases, attributes)

    def __forcedmethodname__(self, method):
        """
        This method builds a forced method. A forced method will
        force all parameters and then call the original method
        on the first argument. The method to use is passed by name.
        """

        def wrapped_method(self, *args):
            # result = force(self)
            # meth = getattr(result, method)
            args = [force(arg) for arg in args]
            # return apply(meth, args)

        return wrapped_method

    def __forcedrmethodname__(self, method, alternative):
        """
        This method builds a forced method. A forced method will
        force all parameters and then call the original method
        on the first argument. The method to use is passed by name.
        An alternative method is passed by name that can be used
        when the original method isn't availabe - but with reversed
        arguments. This can only handle binary methods.
        """

        def wrapped_method(self, other):
            self = force(self)
            other = force(other)
            meth = getattr(self, method, None)
            if meth is not None:
                res = meth(other)
                if res is not NotImplemented:
                    return res
            meth = getattr(other, alternative, None)
            if meth is not None:
                res = meth(self)
                if res is not NotImplemented:
                    return res
            return NotImplemented

        return wrapped_method

    def __forcedmethodfunc__(self, func):
        """
        This method builds a forced method that uses some other
        function to accomplish it's goals. It forces all parameters
        and then calls the function on those arguments.
        """

        def wrapped_method(*args):
            args = [force(arg) for arg in args]
            # return apply(func, args)

        return wrapped_method

    def __delayedmethod__(self, func):
        """
        This method builds a delayed method - one that accomplishes
        it's choire by calling some function if itself is forced.
        A class can define a __delayclass__ if it want's to
        override what class is created on delayed functions. The
        default is to create the same class again we are already
        using.
        """

        def wrapped_method(*args, **kw):
            klass = args[0].__class__
            klass = getattr(klass, "__delayclass__", klass)
            return klass(func, args, kw)

        return wrapped_method


class Promise(object):
    """
    The initialization get's the function and it's parameters to
    delay. If this is a promise that is created because of a delayed
    method on a promise, args[0] will be another promise of the same
    class as the current promise and func will be one of (getattr,
    apply, getitem, getslice). This knowledge can be used to optimize
    chains of delayed functions. Method access on promises will be
    factored as one getattr promise followed by one apply promise.
    """

    __metaclass__ = PromiseMetaClass

    def __init__(self, func, args, kw):
        """
        Store the object and name of the attribute for later
        resolving.
        """
        self.__func = func
        self.__args = args
        self.__kw = kw
        self.__result = NoneSoFar

    def __force__(self):
        """
        This method forces the value to be computed and cached
        for future use. All parameters to the call are forced,
        too.
        """

        # if self.__result is NoneSoFar:
        # args = [force(arg) for arg in self.__args]
        # kw = dict([(k, force(v)) for (k, v)
        #        in self.__kw.items()])
        # self.__result = apply(self.__func, args, kw)
        # return self.__result


# I got this from: http://freshmeat.net/projects/lazypy/
class TailPromise(object):
    __metaclass__ = PromiseMetaClass

    def __init__(self, func, args, kw):
        self.__func = func
        self.__args = args
        self.__kw = kw

    def __arginfo__(self):
        return self.__args, self.__kw

    def __func__(self):
        return self.__func

    def __force__(self):
        return self.__func(*self.__args, **self.__kw)


def trampolined(g):
    def func(*args, **kwargs):
        old_trampolining = func.currently_trampolining

        # if this is not the first call, and it is a tail call:
        if func.currently_trampolining != func:
            # Set up the trampoline!
            func.currently_trampolining = func
            while 1:
                res = g(*args, **kwargs)
                if res.__class__ is TailPromise and res.__func__() is g:
                    # A tail recursion!
                    args, kwargs = res.__arginfo__()
                else:
                    func.currently_trampolining = old_trampolining
                    return res
        else:
            return TailPromise(g, args, kwargs)

    func.currently_trampolining = None
    return func


def delayed(function, check_pickle=True):
    """Decorator used to capture the arguments of a function.

    Pass `check_pickle=False` when:

    - performing a possibly repeated check is too costly and has been done
      already once outside of the call to delayed.

    - when used in conjunction `Parallel(backend='threading')`.

    """
    # Try to pickle the input function, to catch the problems early when
    # using with multiprocessing:
    if check_pickle:
        pickle.dumps(function)

    def delayed_function(*args, **kwargs):
        return function, args, kwargs

    try:
        delayed_function = functools.wraps(function)(delayed_function)
    except AttributeError:
        "functools.wraps fails on some callable objects"
    return delayed_function


##==========================={Using threading library}==========================

# from threading import Thread
import queue as Queue


def run_async_threading(func):
    @wraps(func)
    def async_func(*args, **kwargs):
        queue = Queue.Queue()
        t = Thread(target=func, args=(queue,) + args, kwargs=kwargs)
        t.start()
        return queue, t

    return async_func


##-------------{Tests}------------------

# @run_async
# def do_something_else(queue):
#     while True:
#         sleep(1)
#         print("doing something else")
#         # check if something in the queue and return if so
#         try:
#             queue.get(False)
#         except Queue.Empty:
#             pass
#         else:
#             print("Told to quit")
#             return
#
#
# @run_async
# def print_somedata(queue):
#     print('starting print_somedata')
#     sleep(2)
#     try:
#         queue.get(False)
#     except Queue.Empty:
#         pass
#     else:
#         print("Told to quit")
#         return
#
#     print('print_somedata: 2 sec passed')
#     sleep(2)
#     try:
#         queue.get(False)
#     except Queue.Empty:
#         pass
#     else:
#         print("Told to quit")
#         return
#
#     print('print_somedata: another 2 sec passed')
#     sleep(2)
#     try:
#         queue.get(False)
#     except Queue.Empty:
#         pass
#     else:
#         print("Told to quit")
#         return
#
#     print('finished print_somedata')
#
#
# def test():
#     threads = list()
#
#     # at this moment the thread is created and starts immediately
#     threads.append(print_somedata())
#     print('back in main')
#
#     # at this moment the thread is created and starts immediately
#     threads.append(print_somedata())
#     print('back in main')
#
#     # at this moment the hread is created and starts immediately
#     threads.append(do_something_else())
#     print('back in main')
#
#     # at this moment the hread is created and starts immediately
#     threads.append(do_something_else())
#     print('back in main')
#
#     print("Wait a bit in the main")
#     sleep(1)  # uncomment the wait here to stop the threads very fast ;)
#
#     # you don't have to wait explicitly, as the threads are already
#     # running. This is just an example to show how to interact with
#     # the threads
#     for queue, t in threads:
#         print("Tell thread to stop: %s", t)
#         queue.put('stop')
#         #t.join()


import sys


def reraise(tp, value, tb=None):
    if value.__traceback__ is not tb:
        raise value.with_traceback(tb)
    raise value


def convert_exception(from_exception, to_exception, *to_args, **to_kw):
    """
    Decorator: Catch exception ``from_exception`` and instead raise ``to_exception(*to_args, **to_kw)``.
    Useful when modules you're using in a method throw their own errors that you want to
    convert to your own exceptions that you handle higher in the stack.
    Example: ::
        class FooError(Exception):
            pass
        class BarError(Exception):
            def __init__(self, message):
                self.message = message
        @convert_exception(FooError, BarError, message='bar')
        def throw_foo():
            raise FooError('foo')
        try:
            throw_foo()
        except BarError as e:
            assert e.message == 'bar'
    """

    def wrapper(fn):
        def fn_new(*args, **kw):
            try:
                return fn(*args, **kw)
            except from_exception:
                new_exception = to_exception(*to_args, **to_kw)
                traceback = sys.exc_info()[2]
                value = new_exception
                reraise(new_exception, value, traceback)

        fn_new.__doc__ = fn.__doc__
        return fn_new

    return wrapper
