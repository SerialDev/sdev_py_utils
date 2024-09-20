import types
import inspect
import doctest
import sys


def get_names(f):
    import dis
    import types

    ins = dis.get_instructions(f)
    for x in ins:
        try:
            if (
                x.opcode == 100
                and "<locals>" in next(ins).argval
                and next(ins).opcode == 132
            ):
                yield next(ins).argrepr
                yield from get_names(x.argval)
        except Exception:
            pass


def attach_method(class_instance, name, function):
    if name not in class_instance.__dict__.keys():
        class_instance.__dict__[name] = types.MethodType(function, class_instance)


def nested_attribute(object, attr):
    """
    Get the nested attributes of a pyObj

    Parameters
    ----------

    object : pyObj
       The object to introspect

    attr : str
       The string of the nested attribute to introspect

    Returns
    -------

    None
        nil

    """

    def get_attr(pyobject, attribute_name=""):
        if not attribute_name:
            return pyobject
        else:
            return getattr(pyobject, attribute_name)

    temp = len(inspect_module(object, "attributes"))
    if temp == 0:
        return
    else:
        for i in range(temp):
            attr = inspect_module(object, "attributes")[i][0]
            print(attr, temp)
            # if isinstance(i, dict):
            nested_object = get_attr(object, attr)
            # else:
            #    return
        result = nested_attribute(nested_object, attr)
        return result


def get_attributes(mod):
    """
    Get attributes of a pyObj

    Parameters
    ----------

    mod : pyObj
       Any python object to introspect

    Returns
    -------

    List
        A list of all the attributes of the given {mod}
    """
    attributes = inspect.getmembers(mod, lambda a: not (inspect.isroutine(a)))
    return [
        a
        for a in attributes
        if not (a[0].startswith("__") and a[0].endswith("__") or a[0].startswith("_"))
    ]


def inspect_module(mod, flag=""):
    """
    * Function:
    * Usage: . . .
    * -------------------------------
    * flags: [source, source_lines,  functions, method,
    *  attributes, signature, doc, comments, fun_types
    *  list_functions, parent_module, types, all
    """
    if flag == "source":
        return inspect.getsource(mod).encode().decode("unicode-escape")
    if flag == "source_lines":
        return inspect.getsourcelines(mod)
    if flag == "functions":
        return inspect.getmembers(mod, inspect.isfunction)
    if flag == "func_names":
        return get_function_names(mod)
    if flag == "method":
        return inspect.getmembers(mod, predicate=inspect.ismethod)
    if flag == "method_names":
        return get_method_names(mod)
    if flag == "attributes":
        return get_attributes(mod)
    if flag == "attr_names":
        return get_attribute_names(mod)
    if flag == "signature":
        return inspect.signature(mod)
    if flag == "doc":
        return inspect.getdoc(mod)
    if flag == "comments":
        return inspect.getcomments(mod)
    if flag == "file":
        return inspect.getfile(mod)
    if flag == "doctest":
        return doctest.run_docstring_examples(mod, globals())
    if flag == "list_functions":
        return list_functions(mod)
    if flag == "fun_types":
        return list_functions_types(mod)
    if flag == "parent_module":
        return sys.modules[mod.__module__]
    if flag == "types":
        return inspect_types(mod)
    if flag == "all":
        get_all(mod)


def get_all(mod):
    print("----{functions}-----\n")
    try:
        print(inspect_module(mod, "functions"))
    except Exception as e:
        print(e)
    print("----{method}--------\n")
    try:
        print(inspect_module(mod, "method"))
    except Exception as e:
        print(e)
    print("----{attributes}----\n")
    try:
        print(inspect_module(mod, "attributes"))
    except Exception as e:
        print(e)
    print("----{signature}-----\n")
    try:
        print(inspect_module(mod, "signature"))
    except Exception as e:
        print(e)
    print("----{fun_types}-----\n")
    try:
        print(inspect_module(mod, "fun_types"))
    except Exception as e:
        print(e)
    print("--{parent_module}---\n")
    try:
        print(inspect_module(mod, "parent_module"))
    except Exception as e:
        print(e)
    print("----{types}----------\n")
    try:
        print(inspect_module(mod, "types"))
    except Exception as e:
        print(e)


def inspect_types(mod):
    AsyncGeneratorType, BuiltinFunctionType, BuiltinMethodType, CodeType = [
        [] for i in range(4)
    ]
    CoroutineType, DynamicClassAttribute, FrameType, FunctionType = [
        [] for i in range(4)
    ]
    GeneratorType, GetSetDescriptorType, LambdaType, MappingProxyType = [
        [] for i in range(4)
    ]
    MemberDescriptorType, MethodType, ModuleType, SimpleNamespace = [
        [] for i in range(4)
    ]
    TracebackType = []

    for key, value in mod.__dict__.items():
        if type(value) == types.AsyncGeneratorType:
            AsyncGeneratorType.append((key, value))
        if type(value) == types.BuiltinFunctionType:
            BuiltinFunctionType.append((key, value))
        if type(value) == types.BuiltinMethodType:
            BuiltinMethodType.append((key, value))
        if type(value) == types.CodeType:
            CodeType.append((key, value))
        if type(value) == types.CoroutineType:
            CoroutineType.append((key, value))
        if type(value) == types.DynamicClassAttribute:
            DynamicClassAttribute.append((key, value))
        if type(value) == types.FrameType:
            FrameType.append((key, value))
        if type(value) == types.FunctionType:
            FunctionType.append((key, value))
        if type(value) == types.GeneratorType:
            GeneratorType.append((key, value))
        if type(value) == types.GetSetDescriptorType:
            GetSetDescriptorType.append((key, value))
        if type(value) == types.LambdaType:
            LambdaType.append((key, value))
        if type(value) == types.MappingProxyType:
            MappingProxyType.append((key, value))
        if type(value) == types.MemberDescriptorType:
            MemberDescriptorType.append((key, value))
        if type(value) == types.MethodType:
            MethodType.append((key, value))
        if type(value) == types.ModuleType:
            ModuleType.append((key, value))
        if type(value) == types.SimpleNamespace:
            SimpleNamespace.append((key, value))
        if type(value) == types.TracebackType:
            TracebackType.append((key, value))

    if AsyncGeneratorType:
        print(
            "\n============================={types.AsyncGeneratorType}=========================\n"
        )
        [print(i) for i in AsyncGeneratorType]

    if BuiltinFunctionType:
        print(
            "\n============================={types.BuiltinFunctionType}========================\n"
        )
        [print(i) for i in BuiltinFunctionType]

    if BuiltinMethodType:
        print(
            "\n============================={types.BuiltinMethodType}==========================\n"
        )
        [print(i) for i in BuiltinMethodType]

    if CodeType:
        print(
            "\n============================={types.CodeType}===================================\n"
        )
        [print(i) for i in CodeType]

    if CoroutineType:
        print(
            "\n============================={types.CoroutineType}==============================\n"
        )
        [print(i) for i in CoroutineType]

    if DynamicClassAttribute:
        print(
            "\n============================={types.DynamicClassAttribute}======================\n"
        )
        [print(i) for i in DynamicClassAttribute]

    if FrameType:
        print(
            "\n============================={types.FrameType}==================================\n"
        )
        [print(i) for i in FrameType]

    if FunctionType:
        print(
            "\n============================={types.FunctionType}===============================\n"
        )
        [print(i) for i in FunctionType]

    if GeneratorType:
        print(
            "\n============================={types.GeneratorType}==============================\n"
        )
        [print(i) for i in GeneratorType]

    if GetSetDescriptorType:
        print(
            "\n============================={types.GetSetDescriptorType}=======================\n"
        )
        [print(i) for i in GetSetDescriptorType]

    if LambdaType:
        print(
            "\n============================={types.LambdaType}=================================\n"
        )
        [print(i) for i in LambdaType]

    if MappingProxyType:
        print(
            "\n============================={types.MappingProxyType}===========================\n"
        )
        [print(i) for i in MappingProxyType]

    if MemberDescriptorType:
        print(
            "\n============================={types.MemberDescriptorType}=======================\n"
        )
        [print(i) for i in MemberDescriptorType]

    if MethodType:
        print(
            "\n============================={types.MethodType}=================================\n"
        )
        [print(i) for i in MethodType]

    if ModuleType:
        print(
            "\n============================={types.ModuleType}=================================\n"
        )
        [print(i) for i in ModuleType]

    if SimpleNamespace:
        print(
            "\n============================={types.SimpleNamespace}============================\n"
        )
        [print(i) for i in SimpleNamespace]

    if TracebackType:
        print(
            "\n============================={types.TracebackType}==============================\n"
        )
        [print(i) for i in TracebackType]


def list_functions_types(mod):
    """"""
    func_list = []

    func_list.append(
        [
            mod.__dict__.get(a)
            for a in dir(mod)
            if isinstance(mod.__dict__.get(a), types.FunctionType)
        ]
    )
    return func_list


def list_functions(mod):
    """
    print 'functions in current module:\n', list_functions(sys.modules[__name__])
    print 'functions in inspect module:\n', list_functions(inspect)
    """

    def is_mod_function(mod, func):
        """"""
        return inspect.isfunction(func) and inspect.getmodule(func) == mod

    return [
        func.__name__ for func in mod.__dict__.values() if is_mod_function(mod, func)
    ]


def get_function_names(mod):

    functions = inspect.getmembers(mod, inspect.isfunction)
    names = []
    for x, y in functions:
        names.append(x)
    return names


def get_attribute_names(mod):

    attributes = get_attributes(mod)
    names = []
    for x, y in attributes:
        names.append(x)
    return names


def get_method_names(mod):

    methods = inspect.getmembers(mod, predicate=inspect.ismethod)
    names = []
    for x, y in methods:
        names.append(x)
    return names


def get_arity(fn):
    return len(inspect.getargspec(fn)[0])


def is_subclass(o, bases):
    """
    Similar to the ``issubclass`` builtin, but does not raise a ``TypeError``
    if either ``o`` or ``bases`` is not an instance of ``type``.
    Example::
        >>> is_subclass(IOError, Exception)
        True
        >>> is_subclass(Exception, None)
        False
        >>> is_subclass(None, Exception)
        False
        >>> is_subclass(IOError, (None, Exception))
        True
        >>> is_subclass(Exception, (None, 42))
        False
    """
    try:
        return _issubclass(o, bases)
    except TypeError:
        pass

    if not isinstance(o, type):
        return False
    if not isinstance(bases, tuple):
        return False

    bases = tuple(b for b in bases if isinstance(b, type))
    return _issubclass(o, bases)


def retrieve_name(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    result_name = [k for k, v in locals().items() if v is var][0]

    for fi in reversed(inspect.stack()):
        names = [
            var_name
            for var_name, var_val in fi.frame.f_locals.items()
            if var_val is var
        ]
        if len(names) > 0:
            if len(names) > len(result_name):
                return names[0]
            else:
                return result_name


def source(data):
    import inspect

    result = inspect.getsource(data)
    print(result)
    return result


def find_attributes_with_attribute(
    obj, attribute_name, max_depth, current_depth=0, path=""
):
    """
    * ---------------Function---------------
    * Finds all attributes of an object that have a specific attribute.
    * ----------------Returns---------------
    * -> list[str] : A list of paths to attributes that have the specified attribute.
    * ----------------Params----------------
    * obj : <any> : The object to search for attributes.
    * attribute_name : str : The name of the attribute to search for.
    * max_depth : int : The maximum depth to search for attributes.
    * current_depth : int : The current depth of the search (default: 0).
    * path : str : The current path being searched (default: "").
    * ----------------Usage-----------------
    * This function can be used to find all attributes of an object that have a specific attribute. For example, to find all attributes of an object that have an attribute named "id", you would call `find_attributes_with_attribute(obj, "id", 5)`.
    * ----------------Notes-----------------
    * This function uses recursion to search for attributes, so it may cause a stack overflow if the object has a very deep nested structure.
    """
    matched_attrs = []
    if current_depth > max_depth:
        return matched_attrs
    for attr_name in dir(obj):
        if not attr_name.startswith("__"):  # Ignore built-in attributes/methods
            attr = getattr(obj, attr_name)
            current_path = f"{path}.{attr_name}" if path else attr_name
            if hasattr(attr, attribute_name):
                matched_attrs.append(current_path)
            if hasattr(
                attr, "__dict__"
            ):  # Check if the attribute is an object with its own attributes
                matched_attrs.extend(
                    find_attributes_with_attribute(
                        attr, attribute_name, max_depth, current_depth + 1, current_path
                    )
                )
    return matched_attrs


def find_object_by_class_name(
    obj, target_class_name, max_depth, current_depth=0, path=""
):
    """
    * ---------------Function---------------
    * Searches for objects of a specific class within another object, traversing up to a specified depth.
    * ----------------Returns---------------
    * -> list :: A list of paths to objects of the target class
    * ----------------Params----------------
    * obj :: <any> - The object to search within
    * target_class_name :: str - The name of the class to search for
    * max_depth :: int - The maximum depth to traverse
    * current_depth :: int - (optional, default=0) The current depth of the traversal
    * path :: str - (optional, default="") The current path being traversed
    * ----------------Usage-----------------
    * This function can be used to find all instances of a specific class within an object, such as finding all widgets within a GUI application.
    * Example: `find_object_by_class_name(my_gui, "Widget", 5)` would find all Widget objects within `my_gui` up to a depth of 5.
    * ----------------Notes-----------------
    * This function uses recursion to traverse the object graph, so be careful not to exceed the maximum recursion depth.
    * The function ignores built-in attributes and methods (those that start with double underscore).
    """
    matched_paths = []
    if current_depth > max_depth:
        return matched_paths
    for attr_name in dir(obj):
        if not attr_name.startswith("__"):  # Ignore built-in attributes/methods
            attr = getattr(obj, attr_name)
            current_path = f"{path}.{attr_name}" if path else attr_name
            if attr.__class__.__name__ == target_class_name:
                matched_paths.append(current_path)
            if hasattr(
                attr, "__dict__"
            ):  # Check if the attribute is an object with its own attributes
                matched_paths.extend(
                    find_object_by_class_name(
                        attr,
                        target_class_name,
                        max_depth,
                        current_depth + 1,
                        current_path,
                    )
                )
    return matched_paths


def find_function_definition(function):
    """
    Returns the file path and line number where the given function is defined.

    :param function: The function to find the definition for
    :return: A tuple containing the file path and line number, or None if not found
    """
    try:
        func_code = function.__code__
        file_path = func_code.co_filename
        line_number = func_code.co_firstlineno
        print(f"my_function is defined in {file_path} at line {line_number}")
        return file_path, line_number
    except AttributeError:
        print("Could not find definition for my_function")
        return None

    # WIP
    """

    Layer0 -> Transformer Block <- Attention layer(self_attn)  <- Linear (q_proj) <- Inputdims | Outputdims
       |                |                  |                   <- Linear(k_proj)  <- Inputdims | Outputdims
       |                |                  |                   <- Linear(v_proj)  <- Inputdims | Outputdims
       |                |                  |                   <- Linear(o_proj)  <- Inputdims | Outputdims
       |                |                  |                   <- Rope <- Params
       |                |                  v
       |                |                  MLP  <- Linear(gate_proj) <- Inputdims | Outputdims
       |                |                   |   <- Linear(down_proj) <- Inputdims | Outputdims
       |                |                   |   <- Linear(up_proj) <- Inputdims | Outputdims
       |                |                   v
       |                |        RMSNorm(Input Layernorm)  <- Param
       |                |                   |
       |                |                   v
       |                |        RMSNorm(Post attention Layernorm)  <- Param
       v                v
    Layer1 (...)

    """
