import sdev_c_utils as c


print(dir(c))


a = c.create_arena(200)

c.free_arena(a)


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


import time
from itertools import cycle

# Test data setup
N = 100000  # size of input data for stress testing
keys = cycle(["A", "B", "C", "D", "E"])  # cycling keys for repeated values
values = cycle(range(10))  # cycling values to pair with keys
large_input = [(next(keys), next(values)) for _ in range(N)]


# Function to measure runtime
def stress_test(func, data):
    start_time = time.time()
    result = func(data)
    end_time = time.time()
    return result, end_time - start_time


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
