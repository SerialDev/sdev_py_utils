"""Python functional utilitites"""


# ---{Compose Higher order functions}---#


def compose(*funcs):
    """Return a new function s.t.
       compose(f,g,...)(x) == f(g(...(x)))

    >>> times2 = lambda x: x*2
    >>> minus3 = lambda x: x-3
    >>> mod6 = lambda x: x%6
    >>> f = compose(mod6, times2, minus3)
    >>> all(f(i)==((i-3)*2)%6 for i in range(1000000))
    True

    """

    def inner(data, funcs=funcs):
        result = data
        for f in reversed(funcs):
            result = f(result)
        return result

    return inner
