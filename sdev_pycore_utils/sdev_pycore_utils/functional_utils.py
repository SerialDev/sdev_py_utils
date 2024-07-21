"""Python functional utilitites"""


class Result:
    def __init__(self, value=None, error=None):
        self.value = value
        self.error = error

    def is_ok(self):
        return self.error is None

    def is_err(self):
        return self.error is not None

    def __repr__(self):
        if self.is_ok():
            return f"Ok({self.value})"
        else:
            return f"Err({self.error})"


def try_catch(func, *args, **kwargs):
    """
    * ---------------{Function}---------------
    * try-catch wrapper using the AST library to reduce code noise
    * ----------------{Returns}---------------
    * -> result    ::Result     |Result of the function or Err with exception
    * ----------------{Params}----------------
    * : func       ::callable   |Function to be executed
    * : *args      ::tuple      |Positional arguments for the function
    * : **kwargs   ::dict       |Keyword arguments for the function
    * ----------------{Usage}-----------------
    * def test_function(a, b):
    *     return a / b
    *
    * result = try_catch(test_function, 4, 2)
    * if result.is_ok():
    *     print("Success:", result.value)
    * else:
    *     print("Error:", result.error)
    """
    try:
        return Result(value=func(*args, **kwargs))
    except Exception as e:
        return Result(error=e)


def maybe(func, *args, **kwargs):
    """
    * ---------------{Function}---------------
    * Option monad horrible hack but sometimes why not
    * ----------------{Returns}---------------
    * -> result    ::Result     |Result of the function or Err with exception
    * ----------------{Params}----------------
    * : func       ::callable   |Function to be executed
    * : *args      ::tuple      |Positional arguments for the function
    * : **kwargs   ::dict       |Keyword arguments for the function
    * ----------------{Usage}-----------------
    * def test_function(a, b):
    *     return a / b
    *
    * result = maybe(test_function, 4, 2)
    * if result.is_ok():
    *     print("Success:", result.value)
    * else:
    *     print("Error:", result.error)
    """
    try:
        return Result(value=func(*args, **kwargs))
    except Exception as e:
        return Result(error=e)


def match(value, patterns):
    """
    * ---------------{Function}---------------
    * Works like the 'match' keyword in Rust. Evaluates each pattern function in the 'patterns' list with the 'value' as an argument.
    * If a pattern function returns True, the corresponding action function is called with the 'value' as an argument and the result is returned.
    * If no matching pattern is found, a message is printed indicating that no matching pattern was found and 0 is returned.
    * If an exception occurs while evaluating a pattern-action pair, the error results are printed and 0 is returned.
    * ----------------{Returns}---------------
    * -> result    ::Any        |The result of calling the action function corresponding to the first matching pattern function, or 0 if no matching pattern is found or an exception occurs
    * ----------------{Params}----------------
    * : value      ::Any        |The value to be matched
    * : patterns   ::List[Tuple]|A list of pattern-action pairs, where each pair consists of a pattern function and an action function
    * ----------------{Usage}-----------------
    * def is_even(x):
    *     return x % 2 == 0
    * def double(x):
    *     return x * 2
    * def triple(x):
    *     return x * 3
    * value = 5
    * patterns = [(is_even, double), (lambda x: True, triple)]
    * result = match(value, patterns)
    * print(result)  # Output: 15
    """
    for pattern, action in patterns:
        try:
            if pattern(
                value
            ):  # Check if the pattern function returns True when called with the value
                return action(
                    value
                )  # If the pattern matches, call the action function with the value and return the result
        except (
            Exception
        ) as e:  # If an exception occurs while evaluating the pattern-action pair
            print(
                f"Error occurred in pattern-action pair: {pattern}-{action}"
            )  # Print the pattern and action functions
            print(f"Exception: {e}")  # Print the exception
            exc_type, exc_obj, exc_tb = sys.exc_info()  # Get the exception info
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[
                1
            ]  # Get the file name
            print(
                exc_type, fname, exc_tb.tb_lineno
            )  # Print the exception type, file name, and line number
    print(
        f"No matching pattern found for value: {value}"
    )  # If no matching pattern is found, print a message
    return 0  # Return 0


# ---{Compose Higher order functions}---#


def compose(*funcs):
    """
    * ---------------{Function}---------------
    * Return a new function s.t. compose(f,g,...)(x) == f(g(...(x)))
    * ----------------{Returns}---------------
    * -> result    ::Callable   |The composed function
    * ----------------{Params}----------------
    * : *funcs     ::*Callable  |Functions to be composed
    * ----------------{Usage}-----------------
    * >>> times2 = lambda x: x*2
    * >>> minus3 = lambda x: x-3
    * >>> mod6 = lambda x: x%6
    * >>> f = compose(mod6, times2, minus3)
    * >>> all(f(i)==((i-3)*2)%6 for i in range(1000000))
    * True
    """

    def inner(data, funcs=funcs):
        result = data
        for f in reversed(funcs):
            result = f(result)
        return result

    return inner


def chunker(func, data, chunk_size=1000, *args, **kwargs):
    import time

    results = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i : i + chunk_size]
        try:
            print(
                "\033[33m*"
                + f"Processing chunk {i // chunk_size + 1}/{(len(data) + chunk_size - 1) // chunk_size}"
                + "\033[0m"
            )
            start_time = time.time()
            results.extend(func(chunk, *args, **kwargs))
            elapsed_time = time.time() - start_time
            print(
                "\033[33m*"
                + f"Chunk {i // chunk_size + 1} processed in {elapsed_time:.2f} seconds"
                + "\033[0m"
            )
        except Exception as e:
            print(f"Error: {e}. Issue encountered in chunk {i // chunk_size + 1}.")
    return results
