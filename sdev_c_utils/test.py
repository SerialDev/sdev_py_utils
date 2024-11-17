import sdev_c_utils as c


print(dir(c))


a = c.create_arena(200)

c.free_arena(a)


def stress_test(func, data):
    import time

    start_time = time.time()
    result = func(data)
    end_time = time.time()
    
    total_seconds = end_time - start_time
    execution_time_min = total_seconds / 60
    execution_time_sec = total_seconds % 60
    execution_time_ms = total_seconds * 1000
    execution_time_ns = total_seconds * 1e9

    print(f"\033[33mExecution time:\033[0m")
    print(f"\033[33m- Minutes:\033[0m {execution_time_min:.2f} min")
    print(f"\033[33m- Seconds:\033[0m {execution_time_sec:.2f} sec")
    print(f"\033[33m- Milliseconds:\033[0m {execution_time_ms:.2f} ms")
    print(f"\033[33m- Nanoseconds:\033[0m {execution_time_ns:.2f} ns")
    # print(f"\033[32mResult:\033[0m {result}")

    return result

#

# ------------------------------------------------------------------------- #
#                       VERIFICATION OF FUNCTIONALITY                       #
# ------------------------------------------------------------------------- #


def uniquify_to_dict(value):
    """ """
    result = {}
    temp = []
    current = ""
    for x, y in value:
        if x == current:
            temp.append(y)
        else:
            result[current] = temp
            temp = []
            current = x
            temp.append(y)
        result[current] = temp

    return {k: v for k, v in result.items() if k is not ""}


from itertools import cycle

# Test data setup
N = 100000  # size of input data for stress testing
keys = cycle(["A", "B", "C", "D", "E"])  # cycling keys for repeated values
values = cycle(range(10))  # cycling values to pair with keys
large_input = [(next(keys), next(values)) for _ in range(N)]




# Running the stress test
output, duration = stress_test(uniquify_to_dict, large_input)
print(f"Execution time: {duration} seconds")
print(f"Output size: {len(output)}")  # Number of unique keys in the result
print(output)

output, duration = stress_test(c.uniquify_to_dict, large_input)
print(f"Execution time: {duration} seconds")
print(f"Output size: {len(output)}")  # Number of unique keys in the result
print(output)

# 2x SPEEDUP ACHIEVED!


def flatten_dictionary(d, parent_key="", sep="_"):
    """
    * ---------------{Function}---------------
    * Flatten a nested dictionary into a single-level dictionary
    * ----------------{Returns}---------------
    * -> result    ::Dict       |The flattened dictionary
    * ----------------{Params}----------------
    * : d          ::Dict       |The nested dictionary to flatten
    * : parent_key ::str        |The parent key, used internally for recursion (default: "")
    * : sep        ::str        |The separator to use when concatenating keys (default: "_")
    * ----------------{Usage}-----------------
    * >>> data = {'a': {'b': {'c': 42}}}
    * >>> flatten_dictionary(data)
    * {'a_b_c': 42}
    """
    import collections

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))11111
    return dict(items)


def generate_nested_dict(depth, breadth, value=42):
    nested_dict = value
    for _ in range(depth):
        nested_dict = {f"level_{_}": {f"key_{i}": nested_dict for i in range(breadth)}}
    return nested_dict


# Test data setup
depth = 6  # depth of nested dictionary
breadth = 20  # number of keys at each level
nested_data = generate_nested_dict(depth, breadth)


# Running the stress test
output_python, duration = stress_test(flatten_dictionary, nested_data)
print(f"Execution time: {duration} seconds")
print(f"Output size: {len(output_python)}")  # Number of flattened keys in the result

# Running the stress test
output_c, duration = stress_test(c.flatten_dictionary, nested_data)
print(f"Execution time: {duration} seconds")
print(f"Output size: {len(output_c)}")  # Number of flattened keys in the result


diff = set(output_c.items()) ^ set(output_python.items())




# ------------------------------------------------------------------------- #

import random
def uniquify_list(seq):  # Dave Kirby
    """
    * ---------------{Function}---------------
    * Remove duplicates from a list while preserving the order
    * ----------------{Returns}---------------
    * -> result    ::List       |A new list with duplicates removed
    * ----------------{Params}----------------
    * : seq        ::List       |The input list to remove duplicates from
    * ----------------{Usage}-----------------
    * >>> uniquify_list([1, 2, 2, 3, 4, 4, 5])
    * [1, 2, 3, 4, 5]
    """
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

import numpy as np

N = 1000000  # Size of the list
data = np.random.random(N)


output_python = stress_test(uniquify_list, data)
print(f"Output size: {len(output)}")  # Number of unique elements in the result

output_c = stress_test(c.uniquify_list, list(data))
print(f"Output size: {len(output)}")  # Number of unique elements in the result

diff = set(output_c) ^ set(output_python)
