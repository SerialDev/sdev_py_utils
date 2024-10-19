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


@contextmanager
def temporary_env_vars(**env_vars):
    """
        print("Original PATH:", os.environ.get("PATH"))

    with temporary_env_vars(PATH="/temporary/path"):
        print("Temporary PATH:", os.environ.get("PATH"))

    print("PATH after context:", os.environ.get("PATH"))

    """
    original_env_vars = {key: os.environ.get(key) for key in env_vars}
    try:
        os.environ.update(env_vars)
        yield
    finally:
        for key, value in original_env_vars.items():
            if value is None:
                del os.environ[key]
            else:
                os.environ[key] = value


from contextlib import contextmanager


@contextmanager
def acquire_lock(resource):
    """

    import threading

    lock = threading.Lock()

    with acquire_lock(lock):
        # Critical section
        print("Lock acquired.")

    """
    try:
        resource.acquire()
        yield resource
    finally:
        resource.release()


@contextmanager
def temp_dir():
    """
        import tempfile
    import shutil
    from contextlib import contextmanager

    """
    dirpath = tempfile.mkdtemp()
    try:
        yield dirpath
    finally:
        shutil.rmtree(dirpath)


from contextlib import contextmanager
import time


@contextmanager
def retry_functions(
    functions, retries=3, delay=1, exceptions=(Exception,), args=None, kwargs=None
):
    """
        Context manager to retry a list of functions until one succeeds.

        Parameters
        ----------
        functions : list
            List of functions to attempt.
        retries : int
            Number of retries for each function.
        delay : int or float
            Delay between retries in seconds.
        exceptions : tuple
            Exceptions that trigger a retry.
        args : list or tuple
            Positional arguments to pass to the functions.
        kwargs : dict
            Keyword arguments to pass to the functions.
        USAGE:

    def func1():
        raise ValueError("func1 failed")

    def func2():
        raise RuntimeError("func2 failed")

    def func3():
        return "Success from func3"

    functions = [func1, func2, func3]

    with retry_functions(functions, retries=2, delay=0.5) as result:
        print(f"Result: {result}")

    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    last_exception = None
    for func in functions:
        attempt = 0
        while attempt < retries:
            try:
                result = func(*args, **kwargs)
                yield result
                return  # Exit the context manager upon success
            except exceptions as e:
                attempt += 1
                last_exception = e
                print(
                    f"\033[31mFunction '{func.__name__}' attempt {attempt} failed: {e}\033[0m"
                )
                if attempt < retries:
                    time.sleep(delay)
                else:
                    print(
                        f"\033[31mFunction '{func.__name__}' failed after {retries} retries.\033[0m"
                    )
                    break  # Move to the next function
    # If all functions failed
    print("\033[31mAll functions failed.\033[0m")
    if last_exception:
        raise last_exception


@contextmanager
def atomic_write(filename, mode="w", as_file=True):
    """
    with atomic_write('safe_output.txt') as f:
    f.write("Atomic write content")

    """
    import os
    import tempfile
    from contextlib import contextmanager

    temp_file = tempfile.NamedTemporaryFile(mode=mode, delete=False)
    try:
        yield temp_file if as_file else temp_file.name
        temp_file.close()
        os.replace(temp_file.name, filename)
    except Exception:
        temp_file.close()
        os.unlink(temp_file.name)
        raise


class ResourcePool:
    """
        # Usage

    resources = ['conn1', 'conn2', 'conn3']
    pool = ResourcePool(resources)

    with pool.acquire() as conn:
        print(f"Using {conn}")

    """

    from contextlib import contextmanager
    from queue import Queue

    def __init__(self, resources):
        self._pool = Queue()
        for resource in resources:
            self._pool.put(resource)

    @contextmanager
    def acquire(self):
        resource = self._pool.get()
        try:
            yield resource
        finally:
            self._pool.put(resource)


@contextmanager
def resource_p99(interval=0.1):
    """

    import time
    import requests
    import os

    def simulate_workload():
        # Memory-intensive operation
        data = [i ** 2 for i in range(1000000)]
        time.sleep(0.5)

        # CPU-intensive operation
        for _ in range(5):
            sum([i * i for i in range(1000000)])
            time.sleep(0.2)

        # # Network-intensive operation
        # for _ in range(5):
        #     requests.get('https://www.example.com')
        #     time.sleep(0.2)

        # Disk I/O-intensive operation
        with open('temp_file.bin', 'wb') as f:
            f.write(os.urandom(1024 * 1024 * 10))  # Write 10 MB
        time.sleep(0.5)
        os.remove('temp_file.bin')

    # Use the context manager
    with resource_p99(interval=0.1):
        simulate_workload()

    """

    import tracemalloc
    import psutil
    import time
    import numpy as np
    from contextlib import contextmanager
    import threading

    # Initialize sampling data
    cpu_samples = []
    memory_samples = []
    net_samples_sent = []
    net_samples_recv = []
    disk_samples_read = []
    disk_samples_write = []
    running = True

    def sample_resources():
        prev_net_counters = psutil.net_io_counters()
        prev_disk_counters = psutil.disk_io_counters()
        while running:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_samples.append(cpu_percent)

            # Memory Usage
            current_memory = psutil.Process().memory_info().rss  # Resident Set Size
            memory_samples.append(current_memory)

            # Network I/O
            net_counters = psutil.net_io_counters()
            bytes_sent = net_counters.bytes_sent - prev_net_counters.bytes_sent
            bytes_recv = net_counters.bytes_recv - prev_net_counters.bytes_recv
            net_samples_sent.append(bytes_sent / interval)
            net_samples_recv.append(bytes_recv / interval)
            prev_net_counters = net_counters

            # Disk I/O
            disk_counters = psutil.disk_io_counters()
            read_bytes = disk_counters.read_bytes - prev_disk_counters.read_bytes
            write_bytes = disk_counters.write_bytes - prev_disk_counters.write_bytes
            disk_samples_read.append(read_bytes / interval)
            disk_samples_write.append(write_bytes / interval)
            prev_disk_counters = disk_counters

            time.sleep(interval)

    # Start the sampling thread
    sampler_thread = threading.Thread(target=sample_resources)
    sampler_thread.start()

    try:
        yield
    finally:
        running = False
        sampler_thread.join()

        # Calculate p99 values
        p99_cpu = np.percentile(cpu_samples, 99) if cpu_samples else 0
        p99_memory = np.percentile(memory_samples, 99) if memory_samples else 0
        p99_sent = np.percentile(net_samples_sent, 99) if net_samples_sent else 0
        p99_recv = np.percentile(net_samples_recv, 99) if net_samples_recv else 0
        p99_disk_read = np.percentile(disk_samples_read, 99) if disk_samples_read else 0
        p99_disk_write = (
            np.percentile(disk_samples_write, 99) if disk_samples_write else 0
        )

        # Output results
        print(f"\033[34mCPU p99 usage: {p99_cpu:.2f}%\033[0m")
        print(f"\033[34mMemory p99 usage: {p99_memory / 1024**2:.2f} MB\033[0m")
        print(
            f"\033[34mNetwork p99 usage - Sent: {p99_sent / 1024:.2f} KB/s, Received: {p99_recv / 1024:.2f} KB/s\033[0m"
        )
        print(
            f"\033[34mDisk I/O p99 usage - Read: {p99_disk_read / 1024:.2f} KB/s, Write: {p99_disk_write / 1024:.2f} KB/s\033[0m"
        )
