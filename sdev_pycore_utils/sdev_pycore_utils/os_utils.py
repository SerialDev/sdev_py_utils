""" Python OS std lib utilities"""

import fnmatch
import glob
import gzip
import json
import math
import ntpath
import os
import re
import shutil
import subprocess
from datetime import date, timedelta

import subprocess
import sys


# Define ANSI color codes for output
class Color:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"


import dill as pickle  # Required to pickle lambda functions

# import comtypes.client
# #Generates wrapper for a given library
# def wrap(com_lib):
#     try:
#          comtypes.client.GetModule(com_lib)
#     except:
#          print "Failed to wrap {0}".format(com_lib)

# sys32dir = os.path.join(os.environ["SystemRoot"], "system32")

# #Generate wrappers for all ocx's in system32
# for lib in glob.glob(os.path.join(sys32dir, "*.ocx")):
#     wrap(lib)

# #Generate for all dll's in system32
# for lib in glob.glob(os.path.join(sys32dir, "*.tlb")):
#     wrap(lib)


def checkIfProcessRunning(processName):
    """
    * ---------------{Function}---------------
    * Check if there is any running process that contains the given name processName
    * ----------------{Returns}---------------
    * -> result    ::Bool       |True if the process is running, False otherwise
    * ----------------{Params}----------------
    * : processName ::str        |The name of the process to check for
    * ----------------{Usage}-----------------
    * >>> checkIfProcessRunning('python')
    * True
    """
    import psutil

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            # Check if process name contains the given name string.
            if processName.lower() in proc.name().lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False


def pickle_to_buffer(data):
    """
    * ---------------{Function}---------------
    * Serializes the given data into a BytesIO buffer using pickle.
    * ----------------{Returns}---------------
    * -> buffer    ::BytesIO   |A BytesIO buffer containing the serialized data
    * ----------------{Params}----------------
    * : data       ::Any       |The data to be serialized
    * ----------------{Usage}-----------------
    * >>> data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    * >>> buffer = pickle_to_buffer(data)
    """
    import io

    buffer = io.BytesIO()
    buffer.write(pickle.dumps(data))
    buffer.seek(0)
    return buffer


def b64encode_buffer(buffer):
    """
    * ---------------{Function}---------------
    * Encodes the given BytesIO buffer into base64 format.
    * ----------------{Returns}---------------
    * -> encoded   ::str       |The base64 encoded string of the buffer
    * ----------------{Params}----------------
    * : buffer     ::BytesIO   |The BytesIO buffer to be encoded
    * ----------------{Usage}-----------------
    * >>> data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    * >>> buffer = pickle_to_buffer(data)
    * >>> encoded = b64encode_buffer(buffer)
    """
    import base64

    return base64.b64encode(buffer.read())


def b64decode_data(data):
    """
    * ---------------{Function}---------------
    * Decodes Base64-encoded data into its original form
    * ----------------{Returns}---------------
    * -> decoded_data ::bytes     |The decoded data in bytes
    * ----------------{Params}----------------
    * : data         ::str       |The Base64-encoded data to be decoded
    * ----------------{Usage}-----------------
    * >>> b64decode_data('SGVsbG8gV29ybGQ=')   # 'Hello World' in Base64
    * b'Hello World'
    """
    import base64

    return base64.b64decode(data)


def b64encode_data(data):
    """
    * ---------------{Function}---------------
    * Encodes data in Base64 format
    * ----------------{Returns}---------------
    * -> encoded_data ::bytes    |The Base64-encoded data in bytes
    * ----------------{Params}----------------
    * : data         ::bytes    |The data to be encoded
    * ----------------{Usage}-----------------
    * >>> b64encode_data(b'Hello World')
    * b'SGVsbG8gV29ybGQ='
    """
    import base64

    return base64.b64encode(data)


def load_pickle_from_b64(data):
    """
    * ---------------{Function}---------------
    * Loads a pickle object from base64 encoded data
    * ----------------{Returns}---------------
    * -> pickle_object  ::Any        |The deserialized pickle object
    * ----------------{Params}----------------
    * : data            ::bytes      |The base64 encoded data to be loaded as a pickle object
    * ----------------{Usage}-----------------
    * >>> data = b'gANjYXJyYXlfZGF0YQpTJ3Rlc3QnCnEAKYFxAX1xAihYBgAAAHJvb21zX3Rlc3QKcRQAAABUb2tlbi4=\n'
    * >>> load_pickle_from_b64(data)
    * {'array_data': 'test\n', 'random_test': 'Token.'}
    """
    return pickle.loads(b64decode_data(data))


def cast_bytesio_encoding(data):
    """
    * ---------------{Function}---------------
    * Converts a bytesIO object to a base64 encoded bytes object
    * ----------------{Returns}---------------
    * -> b64_encoded  ::bytes      |The base64 encoded bytes object
    * ----------------{Params}----------------
    * : data         ::BytesIO    |The BytesIO object to be converted to base64 encoding
    * ----------------{Usage}-----------------
    * >>> from io import BytesIO
    * >>> data = BytesIO(b'This is a test string')
    * >>> cast_bytesio_encoding(data)
    * b'VGhpcyBpcyBhIHRlc3Qgc3RyaW5n'
    """
    from base64 import b64encode

    data.seek(0)
    return b64encode(data.read())


def cast_encoding_bytesio(data):
    """
    * ---------------{Function}---------------
    * Converts a base64 encoded bytes object to a bytesIO object
    * ----------------{Returns}---------------
    * -> BytesIO  ::io.BytesIO |The BytesIO object created from the given base64 encoded bytes object
    * ----------------{Params}----------------
    * : data     ::bytes        |The base64 encoded bytes object to be converted to a BytesIO object
    * ----------------{Usage}-----------------
    * >>> from io import BytesIO
    * >>> data = b'VGhpcyBpcyBhIHRlc3Qgc3RyaW5n'
    * >>> cast_encoding_bytesio(data)
    * <_io.BytesIO object at 0x0000021D230A9AD0>
    """
    from base64 import b64decode
    from io import BytesIO

    buf = BytesIO()
    buf.write(b64decode(data))
    buf.seek(0)
    return buf


def path_split_into_list(path):
    """
    * ---------------{Function}---------------
    * Returns all parts of the path as a list, excluding path separators
    * ----------------{Returns}---------------
    * -> parts     ::List[str] |List of path components in the given path, excluding path separators
    * ----------------{Params}----------------
    * : path       ::str        |The path to split into parts
    * ----------------{Usage}-----------------
    * >>> path_split_into_list('/home/user/data/sample.txt')
    * ['home', 'user', 'data', 'sample.txt']
    * >>> path_split_into_list('C:\\Users\\user\\data\\sample.txt')
    * ['C:', 'Users', 'user', 'data', 'sample.txt']
    """
    parts = []
    while True:
        newpath, tail = os.path.split(path)
        if newpath == path:
            assert not tail
            if path and path not in os_path_separators():
                parts.append(path)
            break
        if tail and tail not in os_path_separators():
            parts.append(tail)
        path = newpath
    parts.reverse()
    return parts


def walk_retrieve_filenames(path_to_dir, suffix=".csv"):
    """
    Walk directories and retrieve all filenames matching a filetype

    Parameters
    ----------

    path_to_dir : str
       Path to begin the crawling from

    suffix : str
       suffix of the filetypes to retrieve

    Returns
    -------

    List
        a list of filenames containing {suffix}
    """
    filenames = [
        os.path.join(d, x)
        for d, dirs, files in os.walk(path_to_dir)
        for x in files
        if x.endswith(suffix)
    ]
    return filenames


def move_walked_files(
    path_to_dir=os.getcwd(), destination_dir=os.getcwd() + "//" + "org", suffix=".csv"
):
    """
    Move files matching a suffix to a organization folder

    Parameters
    ----------

    path_to_dir : str
       Path to begin the crawling from

    destination_dir : str
       Path to the destination directory

    suffix : str
       suffix of the filetypes to retrieve and move

    Returns
    -------

    None
    """
    ensure_dir(destination_dir)
    filenames = walk_retrieve_filenames(path_to_dir, suffix)
    for i in filenames:
        name = ntpath.basename(i)
        shutil.move(i, destination_dir + "//" + name)


def extract_basename(path):
    """
    * ---------------{Function}---------------
    * Extracts the basename of a given path. Should work with any OS path on any OS.
    * ----------------{Returns}---------------
    * -> basename  ::str        |The basename extracted from the given path
    * ----------------{Params}----------------
    * : path       ::str        |The path from which the basename should be extracted
    * ----------------{Usage}-----------------
    * >>> extract_basename("/home/user/data/sample.txt")
    * 'sample.txt'
    * >>> extract_basename("C:\\Users\\user\\data\\sample.txt")
    * 'sample.txt'
    """
    basename = re.search(r"[^\\/]+(?=[\\/]?$)", path)
    if basename:
        return basename.group(0)


def os_path_separators():
    """
    * ---------------{Function}---------------
    * Get the path separators used by the operating system
    * ----------------{Returns}---------------
    * -> seps      ::list       |A list of path separators used by the operating system
    * ----------------{Usage}-----------------
    * >>> os_path_separators()
    * ['/']  # Linux and macOS
    * ['\\', '/']  # Windows
    """
    seps = []
    for sep in os.path.sep, os.path.altsep:
        if sep:
            seps.append(sep)
    return seps


def ensure_dir(directory):
    """
    Ensure a directory exists

    Parameters
    ----------

    directory : str
       Name of the directory to check

    Returns
    -------

    None
       nil
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_dir_from_file(file_path):
    """
    Ensure a directory exists in a filepath

    Parameters
    ----------

    file_path : str
       Filepath to check

    Returns
    -------

    None
        nil
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def truncated_path(path, lval=None, rval=None):
    """
    Allow slicing through path for a numpy-like interface

    Parameters
    ----------

    path : str
       Filepath to slice

    lval : int
       left slice

    rval : int
       right slice

    Returns
    -------

    String
       Sliced path
    """
    return "{}".format(os.sep).join(os.getcwd().split(os.sep)[lval:rval])


def to_bytes_io(data):
    from io import BytesIO

    out_buffer = BytesIO()
    out_buffer.write(data)
    out_buffer.seek(0)
    return out_buffer


def to_str_io(data):
    from io import StringIO

    out__buffer = StringIO()
    out_buffer.write(data)
    out_buffer.seek(0)
    return out_buffer


class file_utils(object):
    """
    File manipulation utilities

    Methods
    ----------

    load_file : path|str flag|str
       load a file

    save_file : data|pyObj path|str
       save a file

    to_np : data|pyObj path|str flag|str
       Serialize object to pickle

    to_npz : data|pyObj path|str flag|str
       Serialize multiple np arrays to  to npz file

    from_np : path|str flag|str
       Load a np.array from np or npz file

    to_pickle : data|pyObj path|str flag|str
       Serialize np.array to npz file

    from_pickle : path|str flag|str
       Load a pyObj from a pickle file

    unpickle_iter : path|str flag|str
       Load a pyObj from a pickle file in a streaming fashion

    to_zipped_pickle : data|pyObj path|str flag|str
       Serialize and compress a pickle file

    from_zipped_pickle : path|str flag|str
       Load a pyObj from a compressed pickle file

    to_json : data|pyObj path|str flag|str
       Serialize object to json

    from_json : data|pyObj path|str flag|str
       Serialize object to json

    delete_file : file_path|str
       Delete this file

    delete_empty_folder : dir_path|str
       remove emptry directory

    delete_directory_with_contents : dir_path|str
       remove a directory and all its contents

    Returns
    -------

    None
        nil
    """

    @staticmethod
    def load_file(path, flag="rb"):
        with open(path, flag) as f:
            temp = f.read()
        return temp

    @staticmethod
    def save_file(data, path, flag="wb"):
        with open(path, flag) as f:
            f.write(data)
        return True

    @staticmethod
    def to_np(data, path, flag="wb"):
        import numpy as np

        with open(path, flag) as f:
            np.save(f, data, allow_pickle=True, fix_imports=True)
        return True

    @staticmethod
    def to_npz(data, path, *args, flag="wb"):
        import numpy as np

        with open(path, flag) as f:
            np.savez(f, data, *args)
        return True

    @staticmethod
    def from_np(path, flag="rb"):
        import numpy as np

        with open(path, flag) as f:
            temp = np.load(f)
        return temp

    @staticmethod
    def to_pickle(data, path, flag="wb"):
        with open(path, flag) as f:
            pickle.dump(data, f)

    @staticmethod
    def from_pickle(path, flag="rb"):
        with open(path, flag) as f:
            temp = pickle.load(f)
        return temp

    @staticmethod
    def unpickle_iter(path, flag="rb"):
        import cPickle

        with open(path, flag) as file:
            while file.peek(1):
                yield cPickle.load(file)

    @staticmethod
    def to_zipped_pickle(obj, filename, protocol=-1):
        with gzip.open(filename, "wb") as f:
            pickle.dump(obj, f, protocol)

    @staticmethod
    def from_zipped_pickle(path):
        try:
            with gzip.open(path, "rb") as f:
                loaded_object = pickle.load(f)
                return loaded_object
        except IOError:
            print("Warning: IO Error returning empty dict.")
            return dict()

    @staticmethod
    def to_json(data, path, flag="wb"):
        with open(path, flag) as f:
            json.dump(data, f)

    @staticmethod
    def from_json(path, flag="rb"):
        with open(path, flag) as f:
            temp = json.load(f)
        return temp

    @staticmethod
    def delete_file(file_path):
        os.remove(file_path)
        return None

    @staticmethod
    def delete_empty_folder(dir_path):
        os.rmdir(dir_path)
        return None

    @staticmethod
    def delete_directory_with_contents(dir_path):
        shutil.rmtree(dir_path)
        return None


def filter_by_pattern(input, patterns):
    """
    Filter iterator based on Unix wildcard patterns

    Parameters
    ----------

    input : iterable
       Input to filter

    patterns : str
       patterns to use to filter iterable

    Returns
    -------

    Set
         a set containing all filtered output
    """
    if patterns is None:
        return input
    output = set()
    for pattern in patterns:
        output.update(fnmatch.filter(input, pattern))
    return output


def find_files(directory, patterns=None):
    """
    Find files in a directory based on Unix Patterns

    Parameters
    ----------

    directory : str
       Path to begin crawling from

    patterns : list
       patterns ot use to find files

    Returns
    -------

    Lazy str
         Yields a filename of each file found lazily
    """
    for root, dirnames, filenames in os.walk(directory):
        for filename in filter_by_pattern(filenames, patterns):
            yield (os.path.join(root, filename))


def find_filenames(path_to_dir, suffix=".csv"):
    """
    Find filenames matching a suffix

    Parameters
    ----------

    path_to_dir : str
       Path to begin crawling from

    suffix : str
       suffix of the filenames to retrieve

    Returns
    -------

    List
        A list of filenames that contain {suffix}
    """
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


def safestat(filename):
    """
    * ---------------{Function}---------------
    * Safely get the status of a file, handling interrupted system calls
    * ----------------{Returns}---------------
    * -> statdata   ::os.stat_result |File status information
    * ----------------{Params}----------------
    * : filename    ::str        |The input file name
    * ----------------{Usage}-----------------
    * >>> safestat("test.txt")
    * os.stat_result(st_mode=33188, st_ino=20525453, st_dev=16777220, st_nlink=1, st_uid=501, st_gid=20, st_size=17, st_atime=1646344213, st_mtime=1646344213, st_ctime=1646344213)
    """
    while True:
        try:
            statdata = os.lstat(filename)
            return statdata
        except IOError as error:
            if error.errno != 4:
                raise


def execute(cmd, working_directory=os.getcwd()):
    """
    * ---------------{Function}---------------
    * Execute a command and return its exit status and output
    * ----------------{Returns}---------------
    * -> result    ::bytes      |The output of the command execution
    * ----------------{Params}----------------
    * : cmd        ::str        |The command to execute
    * : working_directory ::str |The working directory for the command (default is the current working directory)
    * ----------------{Usage}-----------------
    * >>> execute("echo 'Hello, World!'")
    * b'Hello, World!\n'
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


def subdirs(path):
    """
    * ---------------{Function}---------------
    * Yield directory names not starting with '.' under given path
    * ----------------{Returns}---------------
    * -> generator  ::generator  |A generator object yielding directory names
    * ----------------{Params}----------------
    * : path        ::str        |The input path for scanning subdirectories
    * ----------------{Usage}-----------------
    * >>> list(subdirs("/example_directory"))
    * ['/example_directory/subdir1', '/example_directory/subdir2']
    """
    for entry in os.scandir(path):
        if not entry.name.startswith(".") and entry.is_dir():
            yield os.path.join(path, entry.name)


def walklevel(some_dir, level=1):
    """
    Walk through directories restricted to a certain level

    Parameters
    ----------

    some_dir : str
       Path to begin crawling from

    level : int
       Level to stop the crawler in

    Returns
    -------

    Generator
        Generator yielding root
    """
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def path_leaf(path):
    """
    * ---------------{Function}---------------
    * Extract the file or directory name from a given path
    * ----------------{Returns}---------------
    * -> tail       ::str        |File or directory name extracted from the path
    * ----------------{Params}----------------
    * : path        ::str        |The input path string
    * ----------------{Usage}-----------------
    * >>> path_leaf("C:/Users/user/documents/test.txt")
    * 'test.txt'
    """
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def os_path_split_asunder(path, debug=False):
    """
    * ---------------{Function}---------------
    * Split a given path into its components
    * ----------------{Returns}---------------
    * -> parts      ::List[str]  |List of components that make up the path
    * ----------------{Params}----------------
    * : path        ::str        |The input path string
    * : debug       ::bool       |Prints debugging information if set to True (default is False)
    * ----------------{Usage}-----------------
    * >>> os_path_split_asunder("C:/Users/user/documents/test.txt")
    * ['C:', 'Users', 'user', 'documents', 'test.txt']
    """
    parts = []
    while True:
        newpath, tail = os.path.split(path)
        if debug:
            print(repr(path), (newpath, tail))
        if newpath == path:
            assert not tail
            if path:
                parts.append(path)
            break
        parts.append(tail)
        path = newpath
    parts.reverse()
    return parts


def spacedman_parts(path):
    """
    * ---------------{Function}---------------
    * Splits the given path into its components, in a way that is compatible with the spacedman convention
    * ----------------{Returns}---------------
    * -> components ::list |The components of the path, in spacedman convention
    * ----------------{Params}----------------
    * : path ::str |The path to split
    * ----------------{Usage}-----------------
    * >>> spacedman_parts('/path/to/file')
    * ['/', 'path', 'to', 'file']
    * >>> spacedman_parts('C:\path\to\file')
    * ['C:', 'path', 'to', 'file']
    """
    components = []
    while True:
        (path, tail) = os.path.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)


# {Check admin rights}#
if sys.platform != "win32":
    try:
        import os
        import sys
        import subprocess

        import ctypes
        from ctypes.wintypes import HANDLE, BOOL, DWORD, HWND, HINSTANCE, HKEY
        from ctypes import c_ulong, c_char_p, c_int, c_void_p

        PHANDLE = ctypes.POINTER(HANDLE)
        PDWORD = ctypes.POINTER(DWORD)

        GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
        GetCurrentProcess.argtypes = ()
        GetCurrentProcess.restype = HANDLE

        OpenProcessToken = ctypes.windll.kernel32.OpenProcessToken
        OpenProcessToken.argtypes = (HANDLE, DWORD, PHANDLE)
        OpenProcessToken.restype = BOOL

        CloseHandle = ctypes.windll.kernel32.CloseHandle
        CloseHandle.argtypes = (HANDLE,)
        CloseHandle.restype = BOOL

        GetTokenInformation = ctypes.windll.Advapi32.GetTokenInformation
        GetTokenInformation.argtypes = (
            HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            DWORD,
            PDWORD,
        )
        GetTokenInformation.restype = BOOL

        TOKEN_READ = 0x20008
        TokenElevation = 0x14

        class ShellExecuteInfo(ctypes.Structure):
            _fields_ = [
                ("cbSize", DWORD),
                ("fMask", c_ulong),
                ("hwnd", HWND),
                ("lpVerb", c_char_p),
                ("lpFile", c_char_p),
                ("lpParameters", c_char_p),
                ("lpDirectory", c_char_p),
                ("nShow", c_int),
                ("hInstApp", HINSTANCE),
                ("lpIDList", c_void_p),
                ("lpClass", c_char_p),
                ("hKeyClass", HKEY),
                ("dwHotKey", DWORD),
                ("hIcon", HANDLE),
                ("hProcess", HANDLE),
            ]

            def __init__(self, **kw):
                ctypes.Structure.__init__(self)
                self.cbSize = ctypes.sizeof(self)
                for fieldName, fieldValue in kw.items():
                    setattr(self, fieldName, fieldValue)

        PShellExecuteInfo = ctypes.POINTER(ShellExecuteInfo)

        ShellExecuteEx = ctypes.windll.Shell32.ShellExecuteExA
        ShellExecuteEx.argtypes = (PShellExecuteInfo,)
        ShellExecuteEx.restype = BOOL

        WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
        WaitForSingleObject.argtypes = (HANDLE, DWORD)
        WaitForSingleObject.restype = DWORD

        SW_HIDE = 0
        SW_SHOW = 5
        SEE_MASK_NOCLOSEPROCESS = 0x00000040
        INFINITE = -1

        ELEVATE_MARKER = "win32elevate_marker_parameter"

        FreeConsole = ctypes.windll.kernel32.FreeConsole
        FreeConsole.argtypes = ()
        FreeConsole.restype = BOOL

        AttachConsole = ctypes.windll.kernel32.AttachConsole
        AttachConsole.argtypes = (DWORD,)
        AttachConsole.restype = BOOL

        ATTACH_PARENT_PROCESS = -1

        def check_admin_rights_elevated():
            """
            Tells you whether current script already has Administrative rights.
            """
            pid = GetCurrentProcess()
            processToken = HANDLE()
            if not OpenProcessToken(pid, TOKEN_READ, ctypes.byref(processToken)):
                raise ctypes.WinError()
            try:
                elevated, elevatedSize = DWORD(), DWORD()
                if not GetTokenInformation(
                    processToken,
                    TokenElevation,
                    ctypes.byref(elevated),
                    ctypes.sizeof(elevated),
                    ctypes.byref(elevatedSize),
                ):
                    raise ctypes.WinError()
                return bool(elevated)
            finally:
                CloseHandle(processToken)

        def waitAndCloseHandle(processHandle):
            """
            Waits till spawned process finishes and closes the handle for it
            """
            WaitForSingleObject(processHandle, INFINITE)
            CloseHandle(processHandle)

        def elevateAdminRights(waitAndClose=True, reattachConsole=True):
            """
            This will re-run current Python script requesting to elevate administrative rights.
            If waitAndClose is True the process that called elevateAdminRights() will wait till elevated
            process exits and then will quit.
            If waitAndClose is False this function returns None for elevated process and process handle
            for parent process (like POSIX os.fork).
            If reattachConsole is False console of elevated process won't be attached to parent process
            so you won't see any output of it.
            """
            if not check_admin_rights_elevated():
                # this is host process that doesn't have administrative rights
                params = subprocess.list2cmdline(
                    [os.path.abspath(sys.argv[0])] + sys.argv[1:] + [ELEVATE_MARKER]
                )
                executeInfo = ShellExecuteInfo(
                    fMask=SEE_MASK_NOCLOSEPROCESS,
                    hwnd=None,
                    lpVerb="runas",
                    lpFile=sys.executable,
                    lpParameters=params,
                    lpDirectory=None,
                    nShow=SW_HIDE if reattachConsole else SW_SHOW,
                )
                if reattachConsole and not all(
                    stream.isatty() for stream in (sys.stdin, sys.stdout, sys.stderr)
                ):
                    # TODO: some streams were redirected, we need to manually work them
                    # currently just raise an exception
                    raise NotImplementedError(
                        "win32elevate doesn't support elevating scripts with "
                        "redirected input or output"
                    )

                if not ShellExecuteEx(ctypes.byref(executeInfo)):
                    raise ctypes.WinError()
                if waitAndClose:
                    waitAndCloseHandle(executeInfo.hProcess)
                    sys.exit(0)
                else:
                    return executeInfo.hProcess
            else:
                # This is elevated process, either it is launched by host process or user manually
                # elevated the rights for this script. We check it by examining last parameter
                if sys.argv[-1] == ELEVATE_MARKER:
                    # this is script-elevated process, remove the marker
                    del sys.argv[-1]
                    if reattachConsole:
                        # Now attach our elevated console to parent's console.
                        # first we free our own console
                        if not FreeConsole():
                            raise ctypes.WinError()
                        # then we attach to parent process console
                        if not AttachConsole(ATTACH_PARENT_PROCESS):
                            raise ctypes.WinError()

                # indicate we're already running with administrative rights, see docstring
                return None

        # def elevateAdminRun(script_path=__file__):
        #     if not check_admin_rights_elevated():
        #         # this is host process that doesn't have administrative rights
        #         executeInfo = ShellExecuteInfo(fMask=SEE_MASK_NOCLOSEPROCESS, hwnd=None, lpVerb='runas',
        #                                        lpFile=sys.executable, lpParameters=script_path,
        #                                        lpDirectory=None,
        #                                        nShow=SW_HIDE)

        #         if not ShellExecuteEx(ctypes.byref(executeInfo)):
        #             raise ctypes.WinError()
    except Exception:
        print("Wintypes not supported in current OS")

# --{Date utilities}-#


def dates_to_list(days=1, weeks=0, formating="%Y-%m-%d"):
    """
    * ---------------{Function}---------------
    * Generate a list of dates in reverse chronological order from today.
    * ----------------{Returns}---------------
    * -> dates     ::List[str]  |List of formatted date strings
    * ----------------{Params}----------------
    * : days       ::int        |Number of days to include in the list (default is 1)
    * : weeks      ::int        |Number of weeks to include in the list (default is 0)
    * : formating  ::str        |Date format string to use for output (default is "%Y-%m-%d")
    * ----------------{Usage}-----------------
    * >>> dates_to_list()
    * ['2023-04-26']
    * >>> dates_to_list(days=3)
    * ['2023-04-26', '2023-04-25', '2023-04-24', '2023-04-23']
    * >>> dates_to_list(weeks=1)
    * ['2023-04-26', '2023-04-25', '2023-04-24', '2023-04-23', '2023-04-22', '2023-04-21', '2023-04-20', '2023-04-19']
    * >>> dates_to_list(days=2, weeks=1, formating="%m/%d/%Y")
    * ['04/26/2023', '04/25/2023', '04/24/2023', '04/23/2023', '04/22/2023', '04/21/2023', '04/20/2023', '04/19/2023', '04/18/2023']
    """
    days += weeks * 7
    dates = []
    dates.append(date.today().strftime(formating))
    for i in range(days):
        temp = (date.today() - timedelta(days=i)).strftime(formating)
        dates.append(temp)
    return dates


# ---{Open atomic}---#

try:
    replace_func = os.replace
except AttributeError:
    replace_func = os.rename


def _doctest_setup():
    try:
        os.remove("/tmp/open_atomic-example.txt")
    except OSError:
        pass


class open_atomic(object):
    """
    Opens a file for atomic writing by writing to a temporary file, then moving
    the temporary file into place once writing has finished.
    When ``close()`` is called, the temporary file is moved into place,
    overwriting any file which may already exist (except on Windows, see note
    below). If moving the temporary file fails, ``abort()`` will be called *and
    an exception will be raised*.
    If ``abort()`` is called the temporary file will be removed and the
    ``aborted`` attribute will be set to ``True``. No exception will be raised
    if an error is encountered while removing the temporary file; instead, the
    ``abort_error`` attribute will be set to the exception raised by
    ``os.remove`` (note: on Windows, if ``file.close()`` raises an exception,
    ``abort_error`` will be set to that exception; see implementation of
    ``abort()`` for details).
    By default, ``open_atomic`` will put the temporary file in the same
    directory as the target file:
    ``${dirname(target_file)}/.${basename(target_file)}.temp``. See also the
    ``prefix``, ``suffix``, and ``dir`` arguments to ``open_atomic()``. When
    changing these options, remember:
        * The source and the destination must be on the same filesystem,
          otherwise the call to ``os.replace()``/``os.rename()`` may fail (and
          it *will* be much slower than necessary).
        * Using a random temporary name is likely a poor idea, as random names
          will mean it's more likely that temporary files will be left
          abandoned if a process is killed and re-started.
        * The temporary file will be blindly overwritten.
    The ``temp_name`` and ``target_name`` attributes store the temporary
    and target file names, and the ``name`` attribute stores the "current"
    name: if the file is still being written it will store the ``temp_name``,
    and if the temporary file has been moved into place it will store the
    ``target_name``.
    .. note::
        ``open_atomic`` will not work correctly on Windows with Python 2.X or
        Python <= 3.2: the call to ``open_atomic.close()`` will fail when the
        destination file exists (since ``os.rename`` will not overwrite the
        destination file; an exception will be raised and ``abort()`` will be
        called). On Python 3.3 and up ``os.replace`` will be used, which
        will be safe and atomic on both Windows and Unix.
    Example::
        >>> _doctest_setup()
        >>> f = open_atomic("/tmp/open_atomic-example.txt")
        >>> f.temp_name
        '/tmp/.open_atomic-example.txt.temp'
        >>> f.write("Hello, world!") and None
        >>> (os.path.exists(f.target_name), os.path.exists(f.temp_name))
        (False, True)
        >>> f.close()
        >>> os.path.exists("/tmp/open_atomic-example.txt")
        True
    By default, ``open_atomic`` uses the ``open`` builtin, but this behaviour
    can be changed using the ``opener`` argument::
        >>> import io
        >>> f = open_atomic("/tmp/open_atomic-example.txt",
        ...                opener=io.open,
        ...                mode="w+",
        ...                encoding="utf-8")
        >>> some_text = u"\u1234"
        >>> f.write(some_text) and None
        >>> f.seek(0)
        0
        >>> f.read() == some_text
        True
        >>> f.close()
    """

    def __init__(
        self,
        name,
        mode="w",
        prefix=".",
        suffix=".temp",
        dir=None,
        opener=open,
        **open_args,
    ):
        self.target_name = name
        self.temp_name = self._get_temp_name(name, prefix, suffix, dir)
        self.file = opener(self.temp_name, mode, **open_args)
        self.name = self.temp_name
        self.closed = False
        self.aborted = False
        self.abort_error = None

    def _get_temp_name(self, target, prefix, suffix, dir):
        if dir is None:
            dir = os.path.dirname(target)
        return os.path.join(dir, "%s%s%s" % (prefix, os.path.basename(target), suffix))

    def close(self):
        if self.closed:
            return
        try:
            self.file.close()
            replace_func(self.temp_name, self.target_name)
            self.name = self.target_name
        except:
            try:
                self.abort()
            except:
                pass
            raise
        self.closed = True

    def abort(self):
        try:
            if os.name == "nt":
                # Note: Windows can't remove an open file, so sacrifice some
                # safety and close it before deleting it here. This is only a
                # problem if ``.close()`` raises an exception, which it really
                # shouldn't... But it's probably a better idea to be safe.
                self.file.close()
            os.remove(self.temp_name)
        except OSError as e:
            self.abort_error = e
        self.file.close()
        self.closed = True
        self.aborted = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if exc_info[0] is None:
            self.close()
        else:
            self.abort()

    def __getattr__(self, attr):
        return getattr(self.file, attr)


def load_or_create(data, path, force=False):
    """
    * ---------------{Function}---------------
    * Load data from a file or create and save the data to a file if it doesn't exist.
    * ----------------{Returns}---------------
    * -> result    ::Any        |Data loaded from the file or the input data after being saved
    * ----------------{Params}----------------
    * : data       ::Any        |Data to be saved; use None to load data from the file
    * : path       ::str        |Path of the file to load or save
    * : force      ::bool       |Force data to be saved even if the file exists (default is False)
    * ----------------{Usage}-----------------
    * >>> data = {"example": "data"}
    * >>> load_or_create(data, "example_data.pkl")
    * >>> loaded_data = load_or_create(None, "example_data.pkl")
    * >>> load_or_create(data, "example_data.pkl", force=True)
    """
    if force == True:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return data
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return data


def load_or_create_locked(data, path, force=False, timeout=1):
    """
    * ---------------{Function}---------------
    * Load data from a file or create and save the data to a file if it doesn't exist, using a file lock.
    * ----------------{Returns}---------------
    * -> result    ::Any        |Data loaded from the file or the input data after being saved
    * ----------------{Params}----------------
    * : data       ::Any        |Data to be saved; use None to load data from the file
    * : path       ::str        |Path of the file to load or save
    * : force      ::bool       |Force data to be saved even if the file exists (default is False)
    * : timeout    ::int        |Timeout for acquiring the file lock (default is 1 second)
    * ----------------{Usage}-----------------
    * >>> data = {"example": "data"}
    * >>> load_or_create_locked(data, "example_data.pkl")
    * >>> loaded_data = load_or_create_locked(None, "example_data.pkl")
    * >>> load_or_create_locked(data, "example_data.pkl", force=True)
    """
    from filelock import Timeout, FileLock

    lock = FileLock(path + ".lock")

    if force == True:
        with lock.acquire(timeout=timeout):
            open(path, "wb").write(pickle.dumps(data))
        return data
    try:
        with open(path, "rb") as f:
            result = pickle.load(f)
        return result
    except Exception:
        with lock.acquire(timeout=timeout):
            open(path, "wb").write(pickle.dumps(data))
        return data


def file_exists(path):
    """
    * ---------------{Function}---------------
    * Check if a file exists at the given path.
    * ----------------{Returns}---------------
    * -> bool      ::Bool       |True if the file exists, False otherwise
    * ----------------{Params}----------------
    * : path       ::str        |Path of the file to check
    * ----------------{Usage}-----------------
    * >>> exists = file_exists("example.txt")
    * >>> print(exists)
    * True
    """
    return os.path.isfile(path)


def file_is_readable(path):
    """
    * ---------------{Function}---------------
    * Check if a file is readable at the given path.
    * ----------------{Returns}---------------
    * -> bool      ::Bool       |True if the file is readable, False otherwise
    * ----------------{Params}----------------
    * : path       ::str        |Path of the file to check
    * ----------------{Usage}-----------------
    * >>> readable = file_is_readable("example.txt")
    * >>> print(readable)
    * True
    """
    return os.access(path, os.R_OK)


def try_makedir(name):
    """
    * ---------------{Function}---------------
    * Create a directory with the given name if it does not already exist.
    * ----------------{Returns}---------------
    * None
    * ----------------{Params}----------------
    * : name       ::str        |Name of the directory to create
    * ----------------{Usage}-----------------
    * >>> try_makedir("example_directory")
    """
    try:
        os.mkdir(os.path.join(os.getcwd(), name))
    except FileExistsError:
        print("{} folder already exists".format(name))


def local_caching(data, name, force=False):
    """
    * ---------------Function---------------
    * This function is used to cache data locally, it will try to load the data from a pickle file,
    if the file does not exist, it will dump the data to the file.

    * ----------------Returns---------------
    * -> data :: <any>
    * The original data passed in, whether it was loaded from a file or not.

    * ----------------Params----------------
    * data :: <any>
    * The data to be cached.
    * name :: str
    * The name of the cache file.
    * force :: bool (default=False)
    * If true, the data will be dumped to a file even if it exists.

    * ----------------Usage-----------------
    * local_caching(my_data, 'my_cache')

    * ----------------Notes-----------------
    * This function will create a directory named 'cached' in the current working directory if it does not exist.
    * If the file exists and force is False, it will load the data from the file.
    * If the file does not exist, or force is True, it will dump the data to the file.
    """
    name = name + ".plk"
    filepath = os.path.join(os.getcwd(), "cached", name)
    try_makedir("cache")

    if force == True:
        with open(filepath, "wb") as f:
            print("Force dump, dumping..")
            pickle.dump(data, f)
        return data

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            print("{} found loading..".format(name))
        return data
    except FileNotFoundError:
        print("{} was not found, dumping".format(name))

        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return data


def convert_size(size_bytes):
    """
    * ---------------{Function}---------------
    * Convert a given size in bytes to a human-readable size string with unit
    * ----------------{Returns}---------------
    * -> size_str  ::str        |Size in a human-readable format (e.g., '2.53 KB')
    * ----------------{Params}----------------
    * : size_bytes ::int        |Size in bytes to be converted
    * ----------------{Usage}-----------------
    * size_str = convert_size(2590) # '2.53 KB'
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def dump_or_load(directory, name, data=None):
    """
    * ---------------{Function}---------------
    * Load a pickled object from the specified directory and name if data is None,
    * otherwise save the given data as a pickled object
    * ----------------{Returns}---------------
    * -> result    ::Any        |Loaded pickled object or None if an error occurs
    * ----------------{Params}----------------
    * : directory ::str         |Directory where the pickle file is located or will be saved
    * : name      ::str         |Name of the pickle file without the extension
    * : data      ::Any (Optional) |Object to be pickled, or None to load an existing pickle file
    """
    name = name + ".plk"
    filepath = os.path.join(os.getcwd(), directory, name)

    try:
        with open(filepath, "rb") as f:
            result = pickle.load(f)
        return result
    except Exception:
        print("none")


def load_pickle(directory, name):
    """
    * ---------------{Function}---------------
    * Load a pickled object from the specified directory and name
    * ----------------{Returns}---------------
    * -> result    ::Any        |Loaded pickled object or 'Failure' if an error occurs
    * ----------------{Params}----------------
    * : directory ::str         |Directory where the pickle file is located
    * : name      ::str         |Name of the pickle file without the extension
    """
    name = name + ".plk"
    filepath = os.path.join(os.getcwd(), directory, name)
    try:
        with open(filepath, "rb") as f:
            result = pickle.load(f)
    except Exception:
        print("{} does not exist".format(filename))
        result = "Failure"
    return result


def save_pickle(directory, name, data=None):
    """
    * ---------------{Function}---------------
    * Save a pickled object to the specified directory and name
    * ----------------{Returns}---------------
    * -> result    ::str        |'Success' if the object is saved, 'Failure' if an error occurs
    * ----------------{Params}----------------
    * : directory ::str         |Directory where the pickle file will be saved
    * : name      ::str         |Name of the pickle file without the extension
    * : data      ::Any         |Object to be pickled
    """
    name = name + ".plk"
    filepath = os.path.join(os.getcwd(), directory, name)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(f)
        result = "Success"
    except Exception:
        print("{} could not be created".format(filename))
        result = "Failure"
    return result


def dump_or_load_pickle(directory, name, data=None):
    """
    * ---------------{Function}---------------
    * Load or save a pickled object depending on the data parameter
    * ----------------{Returns}---------------
    * -> result    ::Any        |Loaded pickled object, 'Success' if saved, or 'Failure' if an error occurs
    * ----------------{Params}----------------
    * : directory ::str         |Directory where the pickle file is located or will be saved
    * : name      ::str         |Name of the pickle file without the extension
    * : data      ::Any (Optional) |Object to be pickled, or None to load an existing pickle file
    """
    if data == None:
        result = load_pickle(directory, name)
    else:
        result = save_pickle(directory, name, data)
    return result


def install_packages_from_file(filename="requirements.txt"):
    """
    * ---------------Function---------------
    *
    * Installs Python packages listed in a requirements.txt file.
    *
    * ----------------Returns---------------
    *
    * -> None, prints output to console.
    *
    * ----------------Params----------------
    *
    * filename :: str, optional
    *
    * The name of the requirements file. Default value is 'requirements.txt'.
    *
    * ----------------Usage-----------------
    *
    * To use the function, simply call it with optional filename parameter.
    *
    * Example 1: To install packages listed in requirements.txt in the same directory:
    *
    * install_packages_from_file()
    *
    * Example 2: To install packages listed in custom_requirements.txt in a different
    * directory:
    *
    * install_packages_from_file(filename='/path/to/custom_requirements.txt')
    """
    print(f"{Color.YELLOW}Starting the installation process...{Color.RESET}")

    # Ensure that pip is up-to-date
    print(f"{Color.YELLOW}Updating pip to the latest version...{Color.RESET}")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print(
                f"{Color.GREEN}Pip has been updated successfully.{Color.RESET}\n{result.stdout}"
            )
        else:
            print(
                f"{Color.RED}Failed to update pip. Error:{Color.RESET}\n{result.stderr}"
            )
            return  # Exit if pip cannot be updated
    except subprocess.CalledProcessError as e:
        print(f"{Color.RED}Failed to update pip. Exception: {str(e)}{Color.RESET}")
        return  # Exit if pip update fails due to an exception

    # Reading and installing packages from the requirements.txt file
    print(
        f"{Color.YELLOW}Reading packages from {filename} and starting installation...{Color.RESET}"
    )
    try:
        with open(filename, "r") as file:
            packages = file.readlines()
        print(f"{Color.YELLOW}Total packages found: {len(packages)}{Color.RESET}")

        for package in packages:
            package = package.strip()
            if package:  # Ensure it's not an empty line
                print(f"{Color.YELLOW}Installing package: {package}...{Color.RESET}")
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--user",
                        package,
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode == 0:
                    print(
                        f"{Color.GREEN}Package '{package}' installed successfully.{Color.RESET}"
                    )
                else:
                    print(
                        f"{Color.RED}Failed to install '{package}'. Error:{Color.RESET}\n{result.stderr}"
                    )
    except FileNotFoundError:
        print(f"{Color.RED}Error: The file '{filename}' does not exist.{Color.RESET}")
    except Exception as e:
        print(f"{Color.RED}An unexpected error occurred: {str(e)}{Color.RESET}")

    print(f"{Color.GREEN}Installation process completed.{Color.RESET}")


def execute_pip_commands(command_strings):
    """
    * ---------------Function---------------
    *
    * execute_pip_commands(command_strings)
    *
    * • Takes a list of pip command strings, parses them into command lists, and
    *   executes them using subprocess.Popen. Enhanced with colorized output to
    *   improve readability and distinguish between successful and failed executions.
    *
    * ----------------Returns---------------
    *
    * • -> None, this function does not explicitly return a value, but it prints
    *   success/failure messages and command output/error details.
    *
    * ----------------Params----------------
    *
    * • command_strings ::list-of-str
    *   • A list of pip command strings to be executed. Each string should be a
    *     valid pip command that can be run in the terminal.
    *
    * ----------------Usage-----------------
    *
    * To use this function, simply pass a list of pip command strings as the
    * command_strings argument. The function will print success/failure messages and
    * command output/error details to the console.
    *
    * Example:
    *
    * ────────────────────
    * execute_pip_commands([
    *     "pip install requests",
    *     "pip install numpy",
    *     "pip install pandas",
    *     "pip install foobar"  # A non-existent package to test failure handling
    * ])
    * special_install_commands = [
    *     "pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 -f https://download.pytorch.org/whl/torch_stable.html",
    *     "pip install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
    * ]
    *
    * execute_pip_commands(special_install_commands)
    * ────────────────────
    *
    * This will execute each pip command in the list and print a message indicating
    * success or failure. If a command is successful, it will also print the command
    * output. If a command fails, it will print error details. The function will print
    * a message for each command specifying its index to improve readability.
    """
    total_commands = len(command_strings)
    print(
        f"{Color.YELLOW}Total commands to execute: {total_commands}{Color.RESET}"
    )  # Yellow color for total commands info

    for index, command in enumerate(command_strings, start=1):
        command_list = command.split()

        # Yellow color for the command being executed
        print(
            f"{Color.YELLOW}Executing command {index} of {total_commands}: {' '.join(command_list)}{Color.RESET}"
        )

        # Use subprocess.Popen to execute the command
        process = subprocess.Popen(
            command_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        # Wait for the command to complete and capture output
        stdout, stderr = process.communicate()

        # Output handling
        if process.returncode == 0:
            # Green color for successful execution
            print(f"{Color.GREEN}Command {index} executed successfully.{Color.RESET}")
            if stdout.strip():
                print(f"{Color.GREEN}Output:\n{stdout}{Color.RESET}")
        else:
            # Red color for failed execution
            print(f"{Color.RED}Failed to execute command {index}.{Color.RESET}")
            if stderr.strip():
                print(f"{Color.RED}Error Details:\n{stderr}{Color.RESET}")
            else:
                print(
                    f"{Color.RED}Error occurred, but no details were provided.{Color.RESET}"
                )


def conditional_import(import_statements):
    """
    * ---------------Function---------------
    * Conditionally imports modules based on the provided import statements
    *
    * ----------------Returns---------------
    * -> None
    *
    * ----------------Params----------------
    * import_statements :: list<str> : A list of import statements, each in the format "module" or "module as alias"
    *
    * ----------------Usage----------------
    *
    * Example usage:
    *
    * import_statements = ["math", "numpy as np"]
    * conditional_import(import_statements)
    *
    * ----------------Notes-----------------
    * This function dynamically imports modules based on the provided import statements.
    * If the module is not already loaded, it will be imported using the __import__ function.
    * The alias is used to assign the imported module to the global namespace.
    """

    import sys

    for statement in import_statements:
        parts = statement.split(" as ")
        module = parts[0]
        alias = parts[1] if len(parts) > 1 else module.split(".")[-1]

        if module not in sys.modules:
            globals()[alias] = __import__(module, fromlist=[""])
        else:
            globals()[alias] = sys.modules[module]


def dict_to_env_vars(env_dict: dict, permanent: bool = False) -> None:
    """
    Set environment variables from a dictionary.

    Args:
    - env_dict: A dictionary with key-value pairs to be set as environment variables.
    - permanent: If True, sets the environment variables in the `~/.zshrc` file, making them persistent across sessions.

    Warning: Environment variables set by this function will only exist for the duration
             of the process that sets them. They will be lost when the process finishes,
             unless the `permanent` option is used.

    Usage:
    >>> env_dict = {'FOO': 'bar', 'BAZ': 'qux'}
    >>> dict_to_env_vars(env_dict)
    >>> os.environ['FOO']
    'bar'
    >>> os.environ['BAZ']
    'qux'

    >>> env_dict = {'FOO': 'bar', 'BAZ': 'qux'}
    >>> dict_to_env_vars(env_dict, permanent=True)
    >>> # Restart your terminal or run `source ~/.zshrc` to apply the changes
    >>> os.environ['FOO']
    'bar'
    >>> os.environ['BAZ']
    'qux'
    """
    import os

    if permanent:
        zshrc_path = os.path.expanduser("~/.zshrc")
        with open(zshrc_path, "a") as f:
            for key, value in env_dict.items():
                f.write(f"export {key}='{value}'\n")
        print(
            f"Environment variables set in {zshrc_path}. Restart your terminal or run `source ~/.zshrc` to apply the changes."
        )
    else:
        for key, value in env_dict.items():
            os.environ[key] = value


def env_vars_to_dict(vars_list: list[str] = None) -> dict:
    """
    Get a list of environment variables and return them as a dictionary.

    Args:
    - vars_list: A list of environment variable names. If None, returns all environment variables.

    Returns:
    - A dictionary with environment variable names as keys and their values as values.

    Usage:
    >>> env_vars_to_dict(['FOO', 'BAZ'])
    {'FOO': 'bar', 'BAZ': 'qux'}

    >>> env_vars_to_dict()
    {'FOO': 'bar', 'BAZ': 'qux', ...}  # returns all environment variables
    """
    import os

    env_dict = {}
    if vars_list is None:
        for key, value in os.environ.items():
            env_dict[key] = value
    else:
        for var in vars_list:
            env_dict[var] = os.environ.get(var, "")
    return env_dict
