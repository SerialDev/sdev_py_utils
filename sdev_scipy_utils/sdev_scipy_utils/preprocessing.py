#

def url_tokenizer(input):
    """
    * type-def ::(str) -> List[str]
    * ---------------{Function}---------------
        * Tokenizes a URL string by splitting it based on "/", ".", and "-" delimiters.
    * ----------------{Returns}---------------
        * : tokens ::List[str] | A list of unique tokens from the input URL string
    * ----------------{Params}----------------
        * : input ::str | The URL string to tokenize
    * ----------------{Usage}-----------------
        * >>> url_tokenizer("https://www.example.com/some-page.html")
        * ["https:", "www", "example", "some", "page", "html"]
    * ----------------{Notes}-----------------
        * This function can be useful for extracting meaningful tokens from a URL for further processing or analysis.
    """
    tokens_by_slash = str(input.encode("utf-8")).split("/")
    all_tokens = set()
    for token_slash in tokens_by_slash:
        tokens = token_slash.split("-")
        for token in tokens:
            tokens_by_dot = token.split(".")
            all_tokens.update(tokens_by_dot)
    all_tokens.discard("com")
    return list(all_tokens)
    # tokensBySlash = str(input.encode("utf-8")).split("/")
    # allTokens = []
    # for i in tokensBySlash:
    #     tokens = str(i).split("-")
    #     tokensByDot = []
    #     for j in range(0, len(tokens)):
    #         tempTokens = str(tokens[j]).split(".")
    #         tokentsByDot = tokensByDot + tempTokens
    #     allTokens = allTokens + tokens + tokensByDot
    # allTokens = list(set(allTokens))
    # if "com" in allTokens:
    #     allTokens.remove("com")
    # return allTokens


def split_keep(s, delimiter):
    """
    * type-def ::(str, str) -> List[str]
    * ---------------{Function}---------------
        * Splits a string by the delimiter, keeping the delimiter at the end of each substring.
    * ----------------{Returns}---------------
        * : split_list ::List[str] | A list of substrings with the delimiter appended
    * ----------------{Params}----------------
        * : s ::str | The string to be split
        * : delimiter ::str | The delimiter to split the string by
    * ----------------{Usage}-----------------
        * >>> split_keep("a,b,c,d", ",")
        * ["a,", "b,", "c,", "d"]
    * ----------------{Notes}-----------------
        * This function can be useful when the delimiter is needed for further processing.
    """
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]


def parse_uri(s):
    """
    * type-def ::(str) -> List[List[List[str]]]
    * ---------------{Function}---------------
        * Parses a URI string, creating a nested list structure based on "/", "?", and "&" delimiters.
    * ----------------{Returns}---------------
        * : parsed_uri ::List[List[List[str]]] | A nested list representing the parsed URI
    * ----------------{Params}----------------
        * : s ::str | The URI string to parse
    * ----------------{Usage}-----------------
        * >>> parse_uri("example.com/foo?bar=1&baz=2")
        * [[["example.com"], ["foo"], ["bar=1", "baz=2"]]]
    * ----------------{Notes}-----------------
        * This function can be used to parse and analyze a URI string more conveniently.
    """
    return [
        [[split_keep(z, "=") for z in split_keep(y, "&")] for y in split_keep(x, "?")]
        for x in s.split("/")
    ]
