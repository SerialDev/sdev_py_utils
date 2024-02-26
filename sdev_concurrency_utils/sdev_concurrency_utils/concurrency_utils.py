import time
import multiprocessing
import types
from multiprocessing import Pool
import ast
from ast import NodeTransformer, copy_location
import inspect
import sys
import numpy as np

# --{Parallel Numpy}-#


def np_parallel(func, data, parts=4):
    def split_array(data, parts=4):
        split_len = np.int(np.ceil(data.shape[0] / parts))
        split_array = []
        for index, i in enumerate(range(parts), 1):
            if index == 1:
                array = data[0:split_len]
                split_array.append((array))
            else:
                array = data[(split_len * (i - 1)) : (split_len * i)]
                split_array.append((array))
        return np.array(split_array)

    def concatenation_string(parts):
        temp = ""
        for i in range(parts):
            temp += "applied[{}], ".format(i)
        return "applied = np.concatenate(({}), axis=0)".format(temp[:-2])

    def np_multithreaded(func, data, parts=4):
        split = split_array(data, parts)
        pool = Pool(parts)
        applied = np.array(pool.map(func, split))
        exec(concatenation_string(parts))
        return applied

    def np_multiprocessing(func, data, parts=4):
        split = split_array(data, parts)
        pool = Pool()
        applied = np.array(pool.map(func, split))
        exec(concatenation_string(parts))
        return applied

    if Pool.__module__ == "multiprocessing.dummy":
        return np_multithreaded(func, data, parts)
    elif Pool.__module__ == "multiprocessing":
        return np_multiprocessing(func, data, parts)


def await_children_done(verbose=True):
    if verbose:
        print("Awaiting for {}".format(multiprocessing.active_children()))
        start_time = time.clock()
        while multiprocessing.active_children():
            time.sleep(1)
        end_time = time.clock()
        print(
            "DONE: waited for {round(end_time - start_time, 3)}".format(
                round(end_time - start_time, 3)
            )
        )
    else:
        while multiprocessing.active_children():
            time.sleep(1)


# --{Ast utilities}--#

def unindent(source_lines):
    """
    * type-def ::(List[str]) -> None
    * ---------------{Function}---------------
        * Unindents a list of source code lines until the first 'def' statement is found.
    * ----------------{Returns}---------------
        * None
    * ----------------{Params}----------------
        * : source_lines ::List[str] | The list of source code lines to unindent.
    * ----------------{Usage}-----------------
        * >>> source_lines = ["    if x > 0:", "        print('Positive')", "    def some_function(x):"]
        * >>> unindent(source_lines)
    * ----------------{Output}----------------
        * The input source_lines will be unindented as follows:
          * ["if x > 0:", "print('Positive')", "def some_function(x):"]
    * ----------------{Dependencies}---------
        * None
    * ----------------{Performance Considerations}----
        * The performance of this function is primarily dependent on the length of the source_lines list.
          * For large lists, consider using more efficient string manipulation techniques or
          * reducing the size of the input list.
    * ----------------{Side Effects}---------
        * This function modifies the input source_lines list.
    * ----------------{Mutability}------------
        * This function modifies the input source_lines list.
    """
    for i, line in enumerate(source_lines):
        source_lines[i] = line.lstrip()
        if source_lines[i][:3] == "def":
            break


def Call(func, args=None, kwargs=None):
    if args is None:
        args = []
    if kwargs is None:
        kwargs = []
    if sys.version_info >= (3, 5):
        return ast.Call(func, args, kwargs)
    else:
        return ast.Call(func, args, kwargs, None, None)


class SchedulerRewriter(NodeTransformer):
    def __init__(self, concurrent_funcs, frameinfo):
        self.arguments = set()
        self.concurrent_funcs = concurrent_funcs
        self.encountered_funcs = set()
        self.line_offset = frameinfo.lineno - 1
        self.filename = frameinfo.filename

    def references_arg(self, node):
        if not isinstance(node, ast.AST):
            return False
        if type(node) is ast.Name:
            return type(node.ctx) is ast.Load and node.id in self.arguments
        for field in node._fields:
            if field == "body":
                continue
            value = getattr(node, field)
            if not hasattr(value, "__iter__"):
                value = [value]
            if any([self.references_arg(child) for child in value]):
                return True
        return False

    def not_implemented_error(self, node, message):
        return NotImplementedError(
            self.filename + "(" + str(node.lineno + self.line_offset) + ") " + message
        )

    @staticmethod
    def top_level_name(node):
        if type(node) is ast.Name:
            return node.id
        elif type(node) is ast.Subscript or type(node) is ast.Attribute:
            return SchedulerRewriter.top_level_name(node.value)
        return None

    def is_concurrent_call(self, node):
        return (
            type(node) is ast.Call
            and type(node.func) is ast.Name
            and node.func.id in self.concurrent_funcs
        )

    def is_valid_assignment(self, node):
        if not (type(node) is ast.Assign and self.is_concurrent_call(node.value)):
            return False
        if len(node.targets) != 1:
            raise self.not_implemented_error(
                node,
                "Concurrent assignment does not support multiple assignment targets",
            )
        if not type(node.targets[0]) is ast.Subscript:
            raise self.not_implemented_error(
                node, "Concurrent assignment only implemented for index based objects"
            )
        return True

    def encounter_call(self, call):
        self.encountered_funcs.add(call.func.id)
        for arg in call.args:
            arg_name = SchedulerRewriter.top_level_name(arg)
            if arg_name is not None:
                self.arguments.add(arg_name)

    def get_waits(self):
        return [
            ast.Expr(
                Call(ast.Attribute(ast.Name(fname, ast.Load()), "wait", ast.Load()))
            )
            for fname in self.encountered_funcs
        ]

    def visit_Call(self, node):
        if self.is_concurrent_call(node):
            raise self.not_implemented_error(
                node, "The usage of the @concurrent function is unsupported"
            )
        node = self.generic_visit(node)
        return node

    def generic_visit(self, node):
        if (isinstance(node, ast.stmt) and self.references_arg(node)) or isinstance(
            node, ast.Return
        ):
            return self.get_waits() + [node]
        return NodeTransformer.generic_visit(self, node)

    def visit_Expr(self, node):
        if type(node.value) is ast.Call:
            call = node.value
            if self.is_concurrent_call(call):
                self.encounter_call(call)
                return node
            elif any([self.is_concurrent_call(arg) for arg in call.args]):
                conc_args = [
                    (i, arg)
                    for i, arg in enumerate(call.args)
                    if self.is_concurrent_call(arg)
                ]
                if len(conc_args) > 1:
                    raise self.not_implemented_error(
                        call,
                        "Functions with multiple @concurrent parameters are unsupported",
                    )
                conc_call = conc_args[0][1]
                self.encounter_call(conc_call)
                call.args[conc_args[0][0]] = ast.Name("__value__", ast.Load())
                if sys.version_info >= (3, 0):
                    args = [ast.arg("__value__", None)]
                else:
                    args = [ast.Name("__value__", ast.Param())]
                call_lambda = ast.Lambda(
                    ast.arguments(
                        args=args, defaults=[], kwonlyargs=[], kw_defaults=[]
                    ),
                    call,
                )
                return copy_location(
                    ast.Expr(
                        ast.Call(
                            func=ast.Attribute(conc_call.func, "call", ast.Load()),
                            args=[call_lambda] + conc_call.args,
                            keywords=[],
                        )
                    ),
                    node,
                )
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if self.is_valid_assignment(node):
            call = node.value
            self.encounter_call(call)
            name = node.targets[0].value
            self.arguments.add(SchedulerRewriter.top_level_name(name))
            index = node.targets[0].slice.value
            call.func = ast.Attribute(call.func, "assign", ast.Load())
            call.args = [ast.Tuple([name, index], ast.Load())] + call.args
            return copy_location(ast.Expr(call), node)
        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        node.decorator_list = []
        node = self.generic_visit(node)
        node.body += self.get_waits()
        return node


# -------{Concurrency decorators}-------#


# -{sync concurrent}-#


class synchronized(object):
    """
    * Function: Decorate a function calling @concurrent functions
    * -----------{returns}------------
    * returns a value for synchronized processes . . .
    * ------------{usage}-------------
    @syncronized
    def func(*args):
        call(concurrent_decorated functions)
    """

    def __init__(self, f):
        callerframerecord = inspect.stack()[1][0]
        info = inspect.getframeinfo(callerframerecord)
        self.frame_info = info
        self.orig_f = f
        self.f = None
        self.ast = None

    def __get__(self, *args):
        raise NotImplementedError(
            "Decorators from deco cannot be used on class methods"
        )

    def __call__(self, *args, **kwargs):
        if self.f is None:
            source = inspect.getsourcelines(self.orig_f)[0]
            unindent(source)
            source = "".join(source)
            self.ast = ast.parse(source)
            rewriter = SchedulerRewriter(concurrent.functions.keys(), self.frame_info)
            rewriter.visit(self.ast.body[0])
            ast.fix_missing_locations(self.ast)
            out = compile(self.ast, "<string>", "exec")
            scope = dict(self.orig_f.__globals__)
            exec(out, scope)
            self.f = scope[self.orig_f.__name__]
        return self.f(*args, **kwargs)


# {concurrent decorator}#


class concurrent(object):
    functions = {}

    @staticmethod
    def custom(constructor=None, apply_async=None):
        @staticmethod
        def _custom_concurrent(*args, **kwargs):
            conc = concurrent(*args, **kwargs)
            if constructor is not None:
                conc.conc_constructor = constructor
            if apply_async is not None:
                conc.apply_async = apply_async
            return conc

        return _custom_concurrent

    def __init__(self, *args, **kwargs):
        self.in_progress = False
        self.conc_args = []
        self.conc_kwargs = {}
        if len(args) > 0 and isinstance(args[0], types.FunctionType):
            self.setFunction(args[0])
        else:
            self.conc_args = args
            self.conc_kwargs = kwargs
        self.results = []
        self.assigns = []
        self.calls = []
        self.arg_proxies = {}
        self.conc_constructor = Pool
        self.apply_async = lambda self, function, args: self.concurrency.apply_async(
            function, args
        )
        self.concurrency = None

    def __get__(self, *args):
        raise NotImplementedError(
            "Decorators from deco cannot be used on class methods"
        )

    def replaceWithProxies(self, args):
        args_iter = args.items() if type(args) is dict else enumerate(args)
        for i, arg in args_iter:
            if type(arg) is dict or type(arg) is list:
                if not id(arg) in self.arg_proxies:
                    self.arg_proxies[id(arg)] = argProxy(id(arg), arg)
                args[i] = self.arg_proxies[id(arg)]

    def setFunction(self, f):
        concurrent.functions[f.__name__] = f
        self.f_name = f.__name__

    def assign(self, target, *args, **kwargs):
        self.assigns.append((target, self(*args, **kwargs)))

    def call(self, target, *args, **kwargs):
        self.calls.append((target, self(*args, **kwargs)))

    def __call__(self, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], types.FunctionType):
            self.setFunction(args[0])
            return self
        self.in_progress = True
        if self.concurrency is None:
            self.concurrency = self.conc_constructor(
                *self.conc_args, **self.conc_kwargs
            )
        args = list(args)
        self.replaceWithProxies(args)
        self.replaceWithProxies(kwargs)
        result = ConcurrentResult(
            self.apply_async(self, concWrapper, [self.f_name, args, kwargs])
        )
        self.results.append(result)
        return result

    def apply_operations(self, ops):
        for arg_id, key, value in ops:
            self.arg_proxies[arg_id].value.__setitem__(key, value)

    def wait(self):
        results = []
        while self.results:
            result, operations = self.results.pop().get()
            self.apply_operations(operations)
            results.append(result)
        for assign in self.assigns:
            assign[0][0][assign[0][1]] = assign[1].result()
        self.assigns = []
        for call in self.calls:
            call[0](call[1].result())
        self.calls = []
        self.arg_proxies = {}
        self.in_progress = False
        return results


# ------{Result}-----#


class ConcurrentResult(object):
    def __init__(self, async_result):
        self.async_result = async_result

    def get(self):
        return self.async_result.get(3e+6)

    def result(self):
        return self.get()[0]


def concWrapper(f, args, kwargs):
    result = concurrent.functions[f](*args, **kwargs)
    operations = [
        inner
        for outer in args + list(kwargs.values())
        if type(outer) is argProxy
        for inner in outer.operations
    ]
    return result, operations


# {Multithreaded Deco}#


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


def run_async(func):
    from threading import Thread
    from functools import wraps
    import queue as Queue

    @wraps(func)
    def async_func(*args, **kwargs):
        queue = Queue.Queue()
        t = Thread(target=func, args=(queue,) + args, kwargs=kwargs)
        t.start()
        return queue, t

    return async_func
