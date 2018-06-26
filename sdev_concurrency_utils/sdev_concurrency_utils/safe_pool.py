"""
A safe Pool implementation
Author : Peter Waller <peter.waller@cern.ch>
Date   : March 2010
Python's built in Pool has the problem that it if a child process dies, that
process is lost from the Pool. This can lead to the exhaustion of the pool and
blocking of the result thread.
"""

from multiprocessing import Manager, current_process, Queue
from multiprocessing.pool import Pool, worker, RUN
from multiprocessing.util import Finalize

from time import sleep


class SafePool(Pool):
    """
    SafePool
    A safe Pool implementation
    Author : Peter Waller <peter.waller@cern.ch>
    Date   : March 2010
    Python's built in Pool has the problem that it if a child process dies, that
    process is lost from the Pool. This can lead to the exhaustion of the pool and
    blocking of the result thread.

    A 'Smarter' Pool which has knowledge of which child processes are working on
    what if they crash.
    """

    def __init__(self, processes=None, initializer=None, initargs=(), polltime=1):

        self.__stopping = False
        self.polltime = polltime

        super(SafePool, self).__init__(processes, initializer, initargs)

        self.initializer = initializer
        self.initargs = initargs

        self.start_monitor_thread()

    def _setup_queues(self):
        """
        Hijack the input and output work queues so that we remember what the
        process was working on in case it crashes.

        This is stored in the `current_work` 'managed' dict
        (see "Manager" in multiprocessing docs)
        """
        # Generate the original queues
        super(SafePool, self)._setup_queues()

        real_get = self._inqueue.get

        def get():
            "A process is about to start working on something. Remember what."
            p = current_process()
            if not hasattr(p, "current_job"):
                print("I am not an augmented worker process. Exiting.")
                raise SystemExit

            work = real_get()
            if work:
                job, i = work[0], work[1] if work[1] is not None else 0
                p.current_job.value, p.current_i.value = job, i

            return work

        self._inqueue.get = get

    def start_monitor_thread(self):
        "Run check_subprocesses in a seperate thread. Kill it gracefully."
        from threading import Thread

        t = Thread(target=self.check_subprocesses)
        t.setDaemon(True)
        t.start()

        def kill_thread(self):
            "Let the child process checker finish"
            self.__stopping = True
            t.join()

        self.__term = Finalize(self, kill_thread, args=(self,), exitpriority=20)

    def start_worker(self):
        """
        Start a new worker thread. Make the worker thread initialize the shared
        current_job variable inside its process.
        """

        from multiprocessing import Value

        current_job, current_i = Value("i"), Value("i")

        def hijacked_initializer(*args):
            p = current_process()
            p.current_job, p.current_i = current_job, current_i
            if callable(self.initializer):
                return self.initializer(*args)

        args = (self._inqueue, self._outqueue, hijacked_initializer, self.initargs)

        p = self.Process(target=worker, args=args)
        p.current_job, p.current_i = current_job, current_i
        p.current_job = current_job
        p.name = p.name.replace("Process", "PoolWorker")
        p.daemon = True
        p.is_from_safepool = True
        p.start()
        self._pool.append(p)

    def check_subprocesses(self):
        """
        Periodically check processes to see if any died. If so, replace them.

        The ideal would be to detect SIGCHLD instead of polling, but
        unfortunately the parent process's get() block seems to prevent this.
        """
        while not self.__stopping:
            # Remove the bad processes from the pool, remember the good ones.

            good_processes = [p for p in self._pool if p.is_alive()]
            bad_processes = [p for p in self._pool if not p.is_alive()]

            self._pool[:] = good_processes  # in place replace

            # We have bad processes. Restart them.
            for p in bad_processes:
                if getattr(p, "is_from_safepool", False):
                    job, i = p.current_job.value, p.current_i.value
                    result = (False, RuntimeError("PID %i Crashed" % p.pid))
                    self._outqueue.put((job, i, result))

                self.start_worker()

            # Wait a little while.
            sleep(self.polltime)
