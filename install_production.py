import os
import subprocess


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def execute(cmd, working_directory=os.getcwd()):
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=working_directory,
    )
    result, error = process.communicate()
    rc = process.returncode

    if rc != 0:
        print(f"\033[31mError: Command failed -> {cmd}\033[0m")
        print(f"\033[31mError Details:\n{error.decode()}\033[0m")
    else:
        print(f"\033[32mSuccess: Command executed -> {cmd}\033[0m")

    return rc, result.decode() if rc == 0 else None


successes = []
failures = []

print("\033[36mStarting production installation mode...\033[0m")
for name, dirs, files in walklevel(os.getcwd(), 1):
    if "setup.py" in files:
        print(f"\033[36mChecking directory: {name}\033[0m")

        rc, result = execute("uv pip install .", name)
        if rc == 0:
            successes.append(name)
            print(f"\033[35mPip install successful in {name}\033[0m")
        else:
            failures.append(name)
            print(f"\033[31mError: Pip install failed in {name}\033[0m")
            rc, result = execute("pip3 install .", name)

        print(
            f"\033[34mOutput:\n{result}\033[0m"
            if result
            else "\033[34mNo output\033[0m"
        )

print("\033[36m\nSummary Report:\033[0m")
print(f"\033[32mSuccesses ({len(successes)}):\033[0m")
for success in successes:
    print(f" - {success}")

print(f"\033[31mFailures ({len(failures)}):\033[0m")
for failure in failures:
    print(f" - {failure}")

if not failures:
    print("\033[32mAll installations were successful.\033[0m")
else:
    print("\033[31mSome installations encountered issues.\033[0m")
