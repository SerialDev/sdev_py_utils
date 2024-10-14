import mlflow
import logging


class TeeOutput:
    import io

    def __init__(self, *streams):
        super().__init__()
        self.streams = streams  # Store the streams (stdout + mlflow buffer)

    def write(self, data):
        # Write to both the original stdout and the mlflow log buffer
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        # Flush all streams
        for stream in self.streams:
            stream.flush()


class MLFRunManager:
    def __init__(self):
        self.run = None

    def __enter__(self):
        if not mlflow.active_run():
            self.run = mlflow.start_run()
            logging.info("Started new run.")
        else:
            self.run = mlflow.active_run()
            logging.info("Using existing active run.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logging.error("Exception occurred: %s", exc_val)
            mlflow.end_run(status="FAILED")
        else:
            logging.info("Run still active.")

    def final_run(self):
        logging.info("Ending the run manually.")
        mlflow.end_run()


def log_stdout_counter():
    count = 0

    def increment():
        nonlocal count
        count += 1
        return count

    return increment


def log_stdout_to_mlflow(strip_ansi=False, end=False):
    from functools import wraps
    from contextlib import redirect_stdout
    import sys
    import io

    counter = log_stdout_counter()  # Create a new counter instance

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            call_count = counter()

            # Get the caller's line number and function name
            func_name = func.__name__
            func_line_number = inspect.getsourcelines(func)[1]

            buffer = io.StringIO()
            original_stdout = sys.stdout
            if strip_ansi:
                ANSI_ESCAPE = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")

            # Automatically enter the mlflow run manager
            with MLFRunManager() as manager:
                # TeeOutput sends output to both console and mlflow buffer
                with redirect_stdout(TeeOutput(original_stdout, buffer)):
                    result = func(*args, **kwargs)

                # Create the folder "stdouts" if it doesn't exist
                folder_name = "stdouts"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

                # Prepare the file name based on line number, function name, and call count
                file_name = (
                    f"{folder_name}/{func_line_number}_{func_name}_{call_count}.txt"
                )

                # Optionally strip ANSI escape codes
                captured_stdout = buffer.getvalue()
                if strip_ansi:
                    captured_stdout = ANSI_ESCAPE.sub("", captured_stdout)

                # Log the captured stdout to a new file in mlflow
                mlflow.log_text(captured_stdout, file_name)
                logging.info(f"Logged captured stdout to {file_name} in mlflow.")
                if end:
                    manager.final_run()

            return result

        return wrapper

    return decorator


def nossl_hf():
    """
    ALERT: Do not use unless you know exactly why I need this.
    """
    import requests
    from huggingface_hub import configure_http_backend

    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.verify = False
        return session

    configure_http_backend(backend_factory=backend_factory)
