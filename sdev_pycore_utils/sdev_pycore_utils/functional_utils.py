"""Python functional utilitites"""

def match(value, patterns):
    """
    Works like the 'match' keyword in Rust. Evaluates each pattern function in the 'patterns' list with the 'value' as an argument.
    If a pattern function returns True, the corresponding action function is called with the 'value' as an argument and the result is returned.
    If no matching pattern is found, a message is printed indicating that no matching pattern was found and 0 is returned.
    If an exception occurs while evaluating a pattern-action pair, the error results are printed and 0 is returned.
    
    Example:
    def is_even(x):
        return x % 2 == 0
    def double(x):
        return x * 2
    def triple(x):
        return x * 3
    value = 5
    patterns = [(is_even, double), (lambda x: True, triple)]
    result = match(value, patterns)
    print(result)  # Output: 15
    
    :param value: The value to be matched
    :type value: Any
    :param patterns: A list of pattern-action pairs, where each pair consists of a pattern function and an action function
    :type patterns: List[Tuple[Callable[[Any], bool], Callable[[Any], Any]]]
    :return: The result of calling the action function corresponding to the first matching pattern function, or 0 if no matching pattern is found or an exception occurs
    :rtype: Any
    """
    for pattern, action in patterns:
        try:
            if pattern(value):  # Check if the pattern function returns True when called with the value
                return action(value)  # If the pattern matches, call the action function with the value and return the result
        except Exception as e:  # If an exception occurs while evaluating the pattern-action pair
            print(f'Error occurred in pattern-action pair: {pattern}-{action}')  # Print the pattern and action functions
            print(f'Exception: {e}')  # Print the exception
            exc_type, exc_obj, exc_tb = sys.exc_info()  # Get the exception info
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]  # Get the file name
            print(exc_type, fname, exc_tb.tb_lineno)  # Print the exception type, file name, and line number
    print(f'No matching pattern found for value: {value}')  # If no matching pattern is found, print a message
    return 0  # Return 0


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
