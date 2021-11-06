def url_tokenizer(input):
    tokensBySlash = str(input.encode("utf-8")).split("/")
    allTokens = []
    for i in tokensBySlash:
        tokens = str(i).split("-")
        tokensByDot = []
        for j in range(0, len(tokens)):
            tempTokens = str(tokens[j]).split(".")
            tokentsByDot = tokensByDot + tempTokens
        allTokens = allTokens + tokens + tokensByDot
    allTokens = list(set(allTokens))
    if "com" in allTokens:
        allTokens.remove("com")
    return allTokens


def split_keep(s, delimiter):
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]


def parse_uri(s):
    return [
        [[split_keep(z, "=") for z in split_keep(y, "&")] for y in split_keep(x, "?")]
        for x in s.split("/")
    ]
