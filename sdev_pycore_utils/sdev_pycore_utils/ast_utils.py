import ast


def try_catch(funcall):
    """
    try-catch wrapper using the AST library to reduce code noise

    Parameters
    ----------

    funcall : str
       String representation of the code to be evaluated

    Returns
    -------

    None
       nil
    """
    try:
        ast.literal_eval(funcall)
    except Exception as e:
        print("{} : failed to execute".format(funcall))


def maybe(expr):
    # Option monad horrible hack but sometimes why not
    try:
        return eval(expr)
    except Exception as e:
        print("exception", e)
        return None
