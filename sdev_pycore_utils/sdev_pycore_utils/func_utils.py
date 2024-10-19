"""Python core functions  utilitites"""

from contextlib import contextmanager
import sys
import time
import os


def print_iter(n, *args, **kwargs):
    """
    * type-def ::String :: *String :: **String >> cout
    * ---------------{Function}---------------
    * print iterators . . .
    * ----------------{Params}----------------
    * : String | name of iterator : *String | Any args to stringify : **String |
    * any kwargs to stringify
    * ----------------{Returns}---------------
    *  >> cout string  . . .
    """
    print("\r[{}] :: {} {}".format(n, args, kwargs), end="")


def apply_at(func, pos, iterable):
    """
    * ---------------{Function}---------------
    * Apply a function on any iterable . . .
    * ----------------{Returns}---------------
    * ->result     ::Generator   |Gen expression
    * ----------------{Params}----------------
    * : func     ::Func        |function
    * : pos      ::Int or List |position to apply
    * : iterable ::Iter        |Any python iterable
    """

    if type(pos) is int:
        return (func(x) if i == pos else x for (i, x) in enumerate(iterable))
    elif type(pos) is list:
        return (func(x) if i in pos_lst else x for (i, x) in enumerate(iterable))


def apply_at_tup(func, pos_lst, iterable, apply_to_value=True):
    """
    * ---------------{Function}---------------
    * Apply a function on any iterable . . .
    * ----------------{Returns}---------------
    * ->result     ::Generator   |Gen expression
    * ----------------{Params}----------------
    * : func     ::Func        |function
    * : pos      ::Int or List |position to apply
    * : iterable ::Iter        |Any python iterable
    """
    temp = []
    if apply_to_value == True:

        if type(pos_lst) is int:
            for i, x in enumerate(iterable):
                if pos_lst == i:
                    temp.append((x[0], func(x[1])))
                else:
                    temp.append(x)
        if type(pos_lst) is list:
            for i, x in enumerate(iterable):
                if i in pos_lst:
                    temp.append((x[0], func(x[1])))
                else:
                    temp.append(x)
    # Apply func at key
    elif apply_to_value == False:
        if type(pos_lst) is int:
            for i, x in enumerate(iterable):
                if pos_lst == i:
                    temp.append((funx(x[0]), x[1]))
                else:
                    temp.append(x)
        if type(pos_lst) is list:
            for i, x in enumerate(iterable):
                if i in pos_lst:
                    temp.append(func(x[0]), x[1])
                else:
                    temp.append(x)
    return temp


def chunked(iterable, chunk_size):
    """
    Split an iterable into chunks of a specified size.

    Parameters
    ----------
    iterable : iterable
        The iterable to split.
    chunk_size : int
        The size of each chunk.

    Returns
    -------
    generator
        A generator yielding chunks.
    """
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(it))
            yield chunk
        except StopIteration:
            if chunk:
                yield chunk
            break


# Context manager to handle logging, processing, and error handling
@contextmanager
def process_chunk_context(chunk_indices, df):
    try:
        temp_df = df.loc[chunk_indices]
        print(
            f"\033[36mStarting processing for chunk with indices: {chunk_indices}\033[0m"
        )  # Section heading
        yield temp_df
        print(
            f"\033[32mSuccessfully processed chunk with indices: {chunk_indices}\033[0m"
        )  # Success completion
    except Exception as e:
        print(
            f"\033[31mError processing chunk with indices: {chunk_indices} - {e}\033[0m"
        )  # Error log
        raise
    finally:
        # print(
        #     f"\033[35mFinished processing chunk with indices: {chunk_indices}\033[0m"
        # )  # Informative note
        pass


# Function that processes chunks using the context manager and yields each chunk
def process_in_chunks(df, chunk_size):
    # Example of using the generator in a loop
    # for df in process_in_chunks(df_full, 10):  # Adjust chunk size as needed

    iterable = df.index
    for chunk_indices in chunked(iterable, chunk_size):
        with process_chunk_context(chunk_indices, df) as temp_df:
            yield temp_df


def batchify(data, batch_size, func):
    """
    Process data in batches using a provided function.

    Parameters
    ----------
    data : iterable
        The data to process.
    batch_size : int
        The size of each batch.
    func : callable
        The function to process each batch.

    Returns
    -------
    list
        List of results from processing each batch.
    """
    results = []
    for batch in chunked(data, batch_size):
        result = func(batch)
        results.extend(result)
    return results


@contextmanager
def time_block(label="Block"):
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"\033[33m{label} executed in {end_time - start_time:.4f} seconds\033[0m")


@contextmanager
def suppress_output():
    """
    print("This will be printed.")

    with suppress_output():
        print("This will NOT be printed.")

    print("This will be printed again.")
    """
    with open(os.devnull, "w") as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


@contextmanager
def resource_monitor(label="Block"):
    """
    Context manager to monitor execution time, CPU usage, and peak memory usage.

    Parameters
    ----------
    label : str
        Label for the code block being monitored.

    USAGE
    import time
    import numpy as np
    def heavy_computation():
        time.sleep(1)
        data = [i**2 for i in range(10)]
        time.sleep(1)
        return sum(data)

    with resource_monitor("Heavy Computation"):
        result = heavy_computation()
        print(f"Result: {result}")

    """

    import time
    import psutil
    import tracemalloc
    from contextlib import contextmanager

    # Initialize process and start monitoring
    process = psutil.Process()
    start_time = time.time()
    start_cpu_times = process.cpu_times()
    start_cpu_percent = psutil.cpu_percent(interval=None)
    tracemalloc.start()

    try:
        yield
    finally:
        # Gather end metrics
        end_time = time.time()
        end_cpu_times = process.cpu_times()
        end_cpu_percent = psutil.cpu_percent(interval=None)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate differences
        elapsed_time = end_time - start_time
        user_cpu_time = end_cpu_times.user - start_cpu_times.user
        system_cpu_time = end_cpu_times.system - start_cpu_times.system
        total_cpu_time = user_cpu_time + system_cpu_time
        cpu_usage_percent = psutil.cpu_percent(interval=None)
        peak_memory_mb = peak / (1024 * 1024)  # Convert bytes to MB

        # Display the results
        print(f"\033[33m{label} executed in {elapsed_time:.4f} seconds\033[0m")
        print(
            f"\033[33mTotal CPU time: {total_cpu_time:.4f} seconds (User: {user_cpu_time:.4f}, System: {system_cpu_time:.4f})\033[0m"
        )
        print(f"\033[33mCPU usage percent: {cpu_usage_percent}%\033[0m")
        print(f"\033[33mPeak memory usage: {peak_memory_mb:.4f} MB\033[0m")


@contextmanager
def detailed_resource_monitor(label="Block", interval=0.5):
    """
        Context manager to monitor execution time, CPU usage, and memory usage in real-time.

        Parameters
        ----------
        label : str
            Label for the code block being monitored.
        interval : float
            Interval in seconds between updates.

    USAGE
    import time
    import numpy as np
    def heavy_computation():
        time.sleep(1)
        data = [i**2 for i in range(10)]
        time.sleep(1)
        return sum(data)

    with detailed_resource_monitor("Detailed Monitoring", interval=0.5):
        result = heavy_computation()
        print(f"Result: {result}")

    """
    import time
    import psutil
    import threading
    import tracemalloc
    from contextlib import contextmanager

    process = psutil.Process()
    start_time = time.time()
    tracemalloc.start()

    # Function to sample and print resource usage
    def sample():
        while not stop_event.is_set():
            cpu_percent = process.cpu_percent(interval=None)
            mem_info = process.memory_info()
            rss_memory = mem_info.rss / (1024 * 1024)  # MB
            current, peak = tracemalloc.get_traced_memory()
            current_memory = current / (1024 * 1024)  # MB
            peak_memory = peak / (1024 * 1024)  # MB

            elapsed_time = time.time() - start_time

            # Use print_iter to output resource usage
            print_iter(
                f"{elapsed_time:.2f}s",
                f"CPU: {cpu_percent:.2f}%",
                f"RSS Memory: {rss_memory:.2f} MB",
                f"Current Mem: {current_memory:.2f} MB",
                f"Peak Mem: {peak_memory:.2f} MB",
            )

            time.sleep(interval)
        print()  # Move to the next line after stopping

    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=sample)
    monitor_thread.start()

    try:
        yield
    finally:
        # Signal the monitoring thread to stop
        stop_event.set()
        monitor_thread.join()

        end_time = time.time()
        tracemalloc.stop()

        elapsed_time = end_time - start_time

        print(f"\n\033[33m{label} executed in {elapsed_time:.4f} seconds\033[0m")
