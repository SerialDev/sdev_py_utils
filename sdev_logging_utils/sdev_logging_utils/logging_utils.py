"""Logging enhancements for python std """

import logging
import traceback
import sys, os, io
from datetime import datetime

should_save_to_file = False
log_filename = ''


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
    print('\r[{}] :: {} {}'.format(n, args, kwargs), end='')


def set_configuration(filename):
    global log_filename, should_save_to_file
    log_filename = filename
    should_save_to_file = True


def Log(log_string):
    log_entry = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M.%S"), log_string)
    print(log_entry)

    if should_save_to_file:
        with open(log_filename, "a") as log_file:
            log_file.write(log_entry + "\n")


def create_logger(logger_name, level):
    """
    * type-def ::str :: int -> logging.logger
    * ---------------{Function}---------------
    * Create a logger with a given name and level to log . . .
    * ----------------{Params}----------------
    * : String | Name of the Logger
    * : Int | Logging level to stdout from
    * ----------------{Returns}---------------
    * logging.Logger Object  . . .
    """
    logger = logging.getLogger(logger_name)

    logger.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def create_log_level(number, name):
    """
    * type-def ::int :: str -> Logger.debugv[injected]
    * ---------------{Function}---------------
    * Create a custom logging level . . .
    * ----------------{Params}----------------
    * : Int | Number to register in logger : String | name to map to the logging
    * level number
    * ----------------{Returns}---------------
    * Injected debug environment for logger with added information . . .
    """
    DEBUG_LEVELV_NUM = number
    logging.addLevelName(DEBUG_LEVELV_NUM, name)
    def debugv(self, message, *args, **kws ):
        if self.isEnabledFor(DEBUG_LEVELV_NUM):
            self._log(DEBUG_LEVELV_NUM, message, args, **kws)
    logging.Logger.debugv = debugv
    return logging.Logger.debugv




def log(logger, level, info):
    """
    * type-def ::logger_object :: int :: string -> logging()
    * ---------------{Function}---------------
    * Log information using a custom logger and level . . .
    * ----------------{Params}----------------
    * : logger_object | logger being used
    * : int | logging level being logged
    * : string | information being logged
    * ----------------{Returns}---------------
    * logging procedure . . .
    """
    return logger.log(level, info)


def log_trace(logger, exception, level=40):
    """
    * type-def ::logger_object :: int :: exception -> logging()
    * ---------------{Function}---------------
    * Log information using a custom logger and level . . .
    * ----------------{Params}----------------
    * : logger_object | logger being used
    * : exception | Exception object to log
    * : int | logging level being logged
    * ----------------{Returns}---------------
    * logging procedure . . .
    """
    return logger.log(level, "{} ::\n{}".format(exception, traceback.print_exc()))


def log_parsed(logger, exception, level=40):
    """
    * type-def ::logger_object :: exception :: int -> logging()
    * ---------------{Function}---------------
    * Parsed exception information using a custom logger and level . . .
    * ----------------{Params}----------------
    * : logger_object | logger being used
    * : exception | Exception object to log
    * : int | logging level being logged
    * ----------------{Returns}---------------
    * logging procedure . . .
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    logger.log(level,
               "Exception type ~ {}  - Filename ~ {} - Line number ~ {} - Exception object ~{}- Exception_traceback ~ {}".format(exc_type, fname, exc_tb.tb_lineno, exc_obj, exc_tb))

class Logging():
    VERBOSE = 1
    DEBUG = 10
    EVENT = 11
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

def register_levels():
    create_log_level(Logging.VERBOSE, 'VERBOSE')
    create_log_level(Logging.EVENT, 'EVENT')


class TeeLogger:
    """
    This class is an attempt to maintain logs of both stdout and stderr for when models are run.
    To use this class, at the beginning of your script insert these lines::
        sys.stdout = TeeLogger("stdout.log", sys.stdout)
        sys.stderr = TeeLogger("stdout.log", sys.stderr)
    """
    def __init__(self, filename: str, terminal: io.TextIOWrapper):
        self.terminal = terminal
        parent_directory = os.path.dirname(filename)
        os.makedirs(parent_directory, exist_ok=True)
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        # We'll special case a particular thing that keras does, to make the log file more
        # readable.  Keras uses ^H characters to get the training line to update for each batch
        # without adding more lines to the terminal output.  Displaying those in a file won't work
        # correctly, so we'll just make sure that each batch shows up on its own line.
        if '\x08' in message:
            message = message.replace('\x08', '')
            if len(message) == 0 or message[-1] != '\n':
                message += '\n'
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


register_levels()
