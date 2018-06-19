
import re


def check_string_word_list(search_list, long_string):
    if re.compile('|'.join(search_list),re.IGNORECASE).search(long_string):
        return True
    else:
        return False

    
def match_pos(string, regex):
    r = re.compile(regex)
    result = []
    for m in r.finditer(string):
        result.append((m.start(), m.group))
    return result


def to_bytes(bytes_or_str):
    if isinstance(bytes_or_str, str):
        value = bytes_or_str.encode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of bytes


def to_str(bytes_or_str):
    if isinstance(bytes_or_str, bytes):
        value = bytes_or_str.decode() # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value # Instance of str

    
