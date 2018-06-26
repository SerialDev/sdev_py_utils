import os, subprocess


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


# Python script to run a command line


def execute(cmd, working_directory=os.getcwd()):
    """
        Purpose  : To execute a command and return exit status
        Argument : cmd - command to execute
        Return   : exit_code
    """
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=working_directory,
    )
    (result, error) = process.communicate()

    rc = process.wait()

    if rc != 0:
        print("Error: failed to execute command:", cmd)
        print(error)
    return result


for name, dirs, files in walklevel(os.getcwd(), 1):
    if "setup.py" in files:
        proc = execute("pip install .", name)
        print(proc)
