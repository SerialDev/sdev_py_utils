from multiprocessing.dummy import Manager
from multiprocessing.managers import BaseProxy
from multiprocessing import cpu_count
import sys
import os
import pdb

import multiprocessing.managers as m


def RebuildProxyNoReferent(func, token, serializer, kwds):
    """
    Function used for unpickling proxy objects.

    If possible the shared object is returned, or otherwise a proxy for it.
    """
    incref = kwds.pop("incref", True) and not getattr(
        process.current_process(), "_inheriting", False
    )
    return func(token, serializer, incref=incref, **kwds)


class Bar(object):
    def __init__(self):
        self.proxy = None

    def set_proxy(self, proxy):
        self.proxy = proxy


class MyProxy(BaseProxy):
    _exposed_ = ("set_proxy",)

    def set_proxy(self, arg):
        self._callmethod("set_proxy", (arg,))

    def __reduce__(self):
        ret = super(MyProxy, self).__reduce__()
        # RebuildProxy is the first item in the ret tuple.
        # So lets replace it, just for our proxy.
        ret = (RebuildProxyNoReferent,) + ret[1:]
        return ret


class ThreadingManager(object):
    """
    Usage:
    tm = ThreadingManager(2, get_detections=cerberus_api.get_detection)
    u = tm.queued_executor('get_detections', [777, 125])

    """

    def __init__(self, cpu_count=cpu_count(), **kwargs):
        self.m = Manager()
        self.pool = self.m.Pool(cpu_count)
        self.kwargs = kwargs
        self.queue = self.m.Queue()
        self.log_queue = self.m.Queue()

    def executor(self, function, *args, **kwargs):
        result = self.pool.map(self.kwargs[function], *args, **kwargs)
        return result

    def queued_executor(self, function, *args, **kwargs):
        result = self.pool.map(self.kwargs[function], *args, **kwargs)
        [self.queue.put(i) for i in result]
        return self.queue

    def async_executor(self, function, *args, **kwargs):
        result = self.pool.map_async(self.kwargs[function], *args, **kwargs)
        sys.stdout.flush()
        return result

    def star_executor(self, function, *args, **kwargs):
        result = self.pool.starmap(self.kwargs[function], *args, **kwargs)
        return result

    def star_queued_executor(self, function, *args, **kwargs):
        result = self.pool.starmap(self.kwargs[function], *args, **kwargs)
        [self.queue.put(i) for i in result]
        return self.queue

    def async_star_executor(self, function, *args, **kwargs):
        result = self.pool.starmap_async(self.kwargs[function], *args, **kwargs)
        return result

    def async_star_queued_executor(self, function, *args, **kwargs):
        result = self.pool.starmap_async(self.kwargs[function], *args, **kwargs)
        [self.queue.put(i) for i in result]
        return self.queue

    def close(self, exception_type, exception_value, traceback):
        self.pool._maintain_pool()
        self.pool.close()
        self.pool.join()


# import concurrent_log


# tm = ThreadingManager(3, tested=concurrent_log.tested, print_log=concurrent_log.print_log)
# # pdb.set_trace()

# tm.async_star_executor('print_log', [['test.log', tm.queue]])

# [tm.async_star_executor('tested', [[2, 'process_{}'.format(i)]]) for i in range(100)]
