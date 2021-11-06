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
    if re.compile("|".join(search_list), re.IGNORECASE).search(long_string):
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
        value = bytes_or_str.encode()  # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value  # Instance of bytes


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
        value = bytes_or_str.decode()  # uses 'utf-8' for encoding
    else:
        value = bytes_or_str
    return value  # Instance of str


# functions to detect/fix double-encoded UTF-8 strings
# Based on http://blogs.perl.org/users/chansen/2010/10/coping-with-double-encoded-utf-8.html
DOUBLE_ENCODED = re.compile(
    """
\xC3 (?: [\x82-\x9F] \xC2 [\x80-\xBF]                                    # U+0080 - U+07FF
       |  \xA0       \xC2 [\xA0-\xBF] \xC2 [\x80-\xBF]                   # U+0800 - U+0FFF
       | [\xA1-\xAC] \xC2 [\x80-\xBF] \xC2 [\x80-\xBF]                   # U+1000 - U+CFFF
       |  \xAD       \xC2 [\x80-\x9F] \xC2 [\x80-\xBF]                   # U+D000 - U+D7FF
       | [\xAE-\xAF] \xC2 [\x80-\xBF] \xC2 [\x80-\xBF]                   # U+E000 - U+FFFF
       |  \xB0       \xC2 [\x90-\xBF] \xC2 [\x80-\xBF] \xC2 [\x80-\xBF]  # U+010000 - U+03FFFF
       | [\xB1-\xB3] \xC2 [\x80-\xBF] \xC2 [\x80-\xBF] \xC2 [\x80-\xBF]  # U+040000 - U+0FFFFF
       |  \xB4       \xC2 [\x80-\x8F] \xC2 [\x80-\xBF] \xC2 [\x80-\xBF]  # U+100000 - U+10FFFF
       )
""",
    re.X,
)


def is_double_encoded(s):
    return DOUBLE_ENCODED.search(s) and True or False


def decode_double_encoded(m):
    s = m.group(0)
    s = re.sub(r"[\xC2-\xC3]", "", s)
    s = re.sub(r"\A(.)", lambda m: chr(0xC0 | (ord(m.group(1)) & 0x3F)), s)
    return s


def fix_double_encoded(s):
    if not is_double_encoded(s):
        return s
    return DOUBLE_ENCODED.sub(decode_double_encoded, s)


from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()


def similarity_string_list(list_to_check):
    # Run  Ratcliff/Obershelp algorithm  between two strings using the built in sequenceMatcher.
    # This has a O(n^2) overhead and would not be suitable in production [same issue with using Levenshtein Distance]
    # Approximate algorithms are better in this case but overkill for a quick cleanup [Eg, Minhash with LSH]
    similars = []
    for i in list_to_check:
        for j in list_to_check:
            similarity = similar(i, j)
            if similarity > 0.8 and similarity < 1.0:
                similars.append((i, j))
    return similars


def serialize_str(data):
    import pickle, base64

    return base64.b64encode(pickle.dumps(data)).decode()


def deserialize_str(data):
    import pickle, base64

    return pickle.loads(base64.b64decode(data))


def snake_casify(letters):
    letters = list(letters)  # strings are immutable in python
    for idx, i in enumerate(letters):
        if idx == 0:
            if i.isupper():
                letters[idx] = i.lower()
        else:
            if i.isupper():
                letters[idx] = "_" + i.lower()
    return "".join(letters)


def any_until_char(character):
    return "(*?){}+".format(character)


def split_keep(s, delimiter):
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]
