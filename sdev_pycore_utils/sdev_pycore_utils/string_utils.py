
import re


def check_string_word_list(search_list, long_string):
    """
    Check a string for a list of words present

    Parameters
    ----------

    search_list : list
       A list of str with words/patterns

    long_string : str
       A string to seach using the search list

    Returns
    -------

    Bool
        Whether there was a amtch or not
    """
    if re.compile('|'.join(search_list),re.IGNORECASE).search(long_string):
        return True
    else:
        return False


def match_pos(string, regex):
    """
    Retrieve positions of regex matches from string

    Parameters
    ----------

    string : str
       String to get the matches from

    regex : str
       Regex patterns to get the matches

    Returns
    -------

    List
        A list of tuples containing the index of match and the match group
    """
    r = re.compile(regex)
    result = []
    for m in r.finditer(string):
        result.append((m.start(), m.group))
    return result


def to_bytes(bytes_or_str):
    """
    Guarantee bytestring from a string or bytestring

    Parameters
    ----------

    bytes_or_str : bytes|str
       A bytestring or string to encode

    Returns
    -------

    Bytes
        Utf-8 encoded bytestring
    """
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of bytes


def to_str(bytes_or_str):
    """
    Guarantee string from a string or bytestring

    Parameters
    ----------

    bytes_or_str : bytes|str
       A bytestring or string to decode

    Returns
    -------

    String
        A decoded string
    """
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of str
