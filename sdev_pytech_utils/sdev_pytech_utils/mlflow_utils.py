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


# Decorator to log stdout to both the console and mlflow
def log_stdout_to_mlflow(func):
    from functools import wraps
    from contextlib import redirect_stdout
    import sys
    import io

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout  # Save the original stdout
        buffer = io.StringIO()  # Buffer to capture output for mlflow

        # TeeOutput sends output to both console and mlflow buffer
        with redirect_stdout(TeeOutput(original_stdout, buffer)):
            result = func(*args, **kwargs)

        # If an active mlflow run exists, log the captured output to mlflow
        if mlflow.active_run():
            captured_stdout = buffer.getvalue()
            mlflow.log_text(captured_stdout, "captured_stdout.txt")
            logging.info("Logged captured stdout to mlflow.")

        return result

    return wrapper
