""" Python OS std lib utilities"""

import fnmatch
import glob
import json
import ntpath
import os
import re
import shutil
import subprocess
import gzip
from datetime import date, timedelta

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


def path_split_into_list(path):
    # Gets all parts of the path as a list, excluding path separators
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
    """Extracts basename of a given path. Should Work with any OS Path on any OS"""
    basename = re.search(r"[^\\/]+(?=[\\/]?$)", path)
    if basename:
        return basename.group(0)


def os_path_separators():
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
        with gzip.open(filename, 'wb') as f:
            pickle.dump(obj, f, protocol)

    @staticmethod
    def from_zipped_pickle(path):
        try:
            with gzip.open(path, 'rb') as f:
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
    """lstat sometimes get Interrupted system calls; wrap it up so we can
    retry"""
    while True:
        try:
            statdata = os.lstat(filename)
            return statdata
        except IOError as error:
            if error.errno != 4:
                raise


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


def subdirs(path):
    """Yield directory names not starting with '.' under given path."""
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
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def os_path_split_asunder(path, debug=False):
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
    components = []
    while True:
        (path, tail) = os.path.split(path)
        if tail == "":
            components.reverse()
            return components
        components.append(tail)


# {Check admin rights}#
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
        **open_args
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
