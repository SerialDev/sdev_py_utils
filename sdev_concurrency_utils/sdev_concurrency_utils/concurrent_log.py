import time
import sys
import traceback
import contextlib, sys

def tested(val, process_name):
    for i in range(100):
            print("{}, {}".format(process_name, time.time()))
    return val * 2


@contextlib.contextmanager
def log_print(file):
    # capture all outputs to a log file while still printing it
    class Logger:
        def __init__(self, file):
            self.terminal = sys.stdout
            self.log = file

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def __getattr__(self, attr):
            return getattr(self.terminal, attr)

    logger = Logger(file)

    _stdout = sys.stdout
    _stderr = sys.stderr
    sys.stdout = logger
    sys.stderr = logger
    yield logger.log
    sys.stdout = _stdout
    sys.stderr = _stderr


def print_log(filename, manager_queue, *args, **kwargs):
    try:
        # with open(filename, "a") as f:
        with log_print(open(filename, 'a')):
            while True:
                m = manager_queue.get()
                print("{}\n".format(m))
                # f.write("{}\n".format(m))

    except Exception as e:
        print(traceback.format_exc())
