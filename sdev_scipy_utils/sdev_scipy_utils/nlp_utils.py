
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
import numpy as np


def get_ngrams(text, n, flag='nltk'):
    n_grams = ngrams(word_tokenize(text), n)
    if flag == 'gensim':
        try:
            return np.array(['_'.join(grams) for grams in n_grams])
        except Exception as e:
            print(e)
            return []
    elif flag == 'nltk':
        try:
            return np.array([' '.join(grams) for grams in n_grams])
        except Exception as e:
            print(e)
            return []


def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])


def rank_ngrams(df_series, n):
    return df_series.apply(pd.Series).stack().groupby(level=0).apply(lambda x: x.drop_duplicates()).value_counts()[:n]


def get_max_len(series):
    length = 0
    for index, i in enumerate(series):
        if len(i) > length:
            length = len(i)
            indexed = index
    return length, indexed


def _w2v_word_model(word, num_features, model):
    """
    * Function:
    * Usage:Lemmatized = Lemmatized.reshape(Lemmatized.shape[0], 1)
    * Lemmatized_w2v = np.apply_along_axis(_w2v_word_model, 1, Lemmatized)  . . .
    * -------------------------------
    * This function returns a word2vec value for words [vectorized]
    *
    """
    empty_word = np.zeros(num_features).astype(float)

    word = word[0]
    if word in w2v_model:
        return w2v_model[word]
    else:
        return empty_word


def extract_ngrams_np(ngrams):
    tmp = []
    for i in ngrams:
        for j in i:
            tmp.append(j)
    return tmp


# from string import punctuation

# def countInFile(filename):
#     with open(filename) as f:
#         linewords = (line.translate(None, punctuation).lower().split() for line in f)
#         return Counter(chain.from_iterable(linewords))


from unidecode import unidecode
import string

all_letters = string.ascii_letters + " .,;'"
all_letters = ''.join([chr(i) for i in range(128)])
all_letters = string.printable

n_letters = len(all_letters)


def unicode_to_ascii(s):
    """
    Turn a Unicode string to plain ASCII
    thanks to http://stackoverflow.com/a/518232/2809427

    Parameters
    ----------

    s : str
       Unicode string to convert

    Returns
    -------

    str
        Ascii representation of the given Unicode Sequence
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters)


def letter_to_index(letter):
    """
    Find letter index from all_letters, e.g. "a" = 0

    Parameters
    ----------

    letter : str
       A letter to match

    Returns
    -------

    int
        index number from all_letters
    """
    return all_letters.find(letter)


def letter_to_tensor(letter):
    """
    Turn a letter into a <1 x n_letters> Tensor

    Parameters
    ----------

    letter : str
       A letter to transform into a tensor

    Returns
    -------

    np.array
        Tensor representation of the letter
    """
    tensor = np.zeros((1, n_letters))
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line, length=None):
    """
    Turn a line into a <line_length x 1 x n_letters>,
    or an array of one-hot letter vectors

    Parameters
    ----------

    line : str
       A line to convert into a tensor

    length : int
       length of the tensor to generate (max line_size ideally)

    Returns
    -------

    np.array
        Tensor representation of the sentence
    """

    if length == None:
        length = len(line)
    tensor = np.zeros((length, 1, n_letters))
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    result = tensor.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])
    return result.reshape(1, length, n_letters)


def word_to_tensor(word, tensor_length=10):
    """
    Turn a word into a <word_length x 1 x n_letters> tensor

    Parameters
    ----------

    word : str
       Word to represent as a tensor

    tensor_length : int
       Length of the output tensor (max word_len ideally)

    Returns
    -------

    np.array
        Tensor representation of the word
    """
    assert len(word) <= tensor_length, "length={} is > than the desired tensor".format(len(word))

    length = len(word)
    word = line_to_tensor(word)
    word = word.reshape(word.shape[0], word.shape[1] * word.shape[2])

    if length < tensor_length:
        result = np.concatenate((word[0], np.zeros(100 * (tensor_length - length))))
        result = result.reshape(1, result.shape[0])
        return result
    return word


def Build_STDM(docs, **kwargs):
    """
    Build Spares Term Document Matrix

    Parameters
    ----------

    docs : np.array
       An array with all the documents to convert to a term document matrix

    kwargs : args
       Arguments to pass to the countvecorizer

    Returns
    -------

    sparsematrix && list
         a sparsematrix and a list of vocabulary keys
    """
    from sklearn.feature_extraction.text import CountVectorizer
    import pandas as pd
    vectorizer = CountVectorizer(**kwargs)
    sparsematrix = vectorizer.fit_transform(docs)
    vocab = vectorizer.vocabulary_.keys()
    return sparsematrix, vocab



# Define a topic mining function (non-negative matrix factorization)
def nmf(M, components=5, iterations=5000):
    # Initialize to matrices
    W = np.asmatrix(np.random.random(([M.shape[0], components])))
    H = np.asmatrix(np.random.random(([components, M.shape[1]])))
    for n in range(0, iterations): 
        H = np.multiply(H, (W.T * M) / (W.T * W * H + 0.001))
        W = np.multiply(W, (M * H.T) / (W * (H * H.T) + 0.001))
        print("%d/%d" % (n, iterations))    # Note 'logging' module
    return (W, H)


# TODO Vectorized numpy implementation of this
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


# From https://en.wikipedia.org/wiki/Burrows%E2%80%93Wheeler_transform
def bwt(s):
    """Apply Burrows-Wheeler transform to input string. Not indicated by a unique byte but use index list"""
    # Table of rotations of string
    table = [s[i:] + s[:i] for i in range(len(s))]
    # Sorted table
    table_sorted = table[:]
    table_sorted.sort()
    # Get index list of ((every string in sorted table)'s next string in unsorted table)'s index in sorted table
    indexlist = []
    for t in table_sorted:
        index1 = table.index(t)
        index1 = index1+1 if index1 < len(s)-1 else 0
        index2 = table_sorted.index(table[index1])
        indexlist.append(index2)
    # Join last characters of each row into string
    r = ''.join([row[-1] for row in table_sorted])
    return r, indexlist


def ibwt(r,indexlist):
    """Inverse Burrows-Wheeler transform. Not indicated by a unique byte but use index list"""
    s = ''
    x = indexlist[0]
    for _ in r:
        s = s + r[x]
        x = indexlist[x]
    return s



def count_vectorizer_vocab(data, nb_top_words=50000):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=nb_top_words - 1,
    )
    train_split_data_features = vectorizer.fit_transform(data)
    # Convert to an array for easier handling instead of a sparse matrix
    train_split_data_features = train_split_data_features.toarray()
    vocab = vectorizer.get_feature_names()
    freq = np.sum(train_split_data_features, axis=0)

    df_vocab = pd.DataFrame(list(zip(vocab, freq)), columns=['vocab', 'freq'])
    # sort by frequency rank
    df_vocab = df_vocab.sort_values(by="freq", ascending=False)
    df_vocab.reset_index(drop=True, inplace=True)
    df_vocab.index = df_vocab.index + 1 # Increase this to make room for null character

    # Invert word/int pairs to get our lookup with word as key#Invert
    vocab_idx = {key:value for (key, value) in zip(df_vocab["vocab"], df_vocab.index)}
    return vocab_idx, df_vocab



def words_to_index(wordlist, vocab=None):
    """Minifunction for pandas.apply(). Replaces each word with respective index.
    If it's not in the vocab, replace with 0"""
    return [vocab[word] if word in vocab else 0 for word in wordlist]



def arithmetize(text,basis=2**16):
    """ convert substring to number using basis powers
    employs Horner rule """
    partial_sum=0
    for ch in text:
        partial_sum = partial_sum*basis+ord(ch) # Horner
    return partial_sum


def arithmetize_text(text,m,basis=2**16):
     """ computes arithmization of all m long substrings
     of text, using basis powers """
     t=[] # will store list of numbers representing
     # consecutive substrings
     for s in range(0,len(text)-m+1):
         t = t+[arithmetize(text[s:s+m],basis)]
         # append the next number to existing t
     return t


#Something is Off here TODO text2 and efficient
def arithmetize_text2(text,m,basis=2**16):
    """ efficiently computes arithmetization of all m long
    substrings of text, using basis powers """
    b_power=basis**(m-1)
    t=[arithmetize(text[0:m],basis)]
    # t[0] equals first word arithmetization
    for s in range(1,len(text)-m+1):
        new_number=(t[s-1]-ord(text[s-1])*b_power)*basis
        +ord(text[s+m-1])
        t=t+[new_number] # append new_number to existing
    return t
    # t stores list of numbers representing m long words of text 

 
def arithmetize_text_efficient(text, m, basis=2**6):
    """
    efficiently computes arithmetization of all m long
    substrings of text, using basis powers
    """

    b_power=basis**(m-1)
    t=[arithmetize(text[0:m], basis)]
    # t[0] equals first word arithmetization
    for s in range(1, len(text) -m +1):
        new_number = (t[s-1] - ord(text[s-1]) * b_power) * basis + ord(text[s+m-1])
        t = t + [new_number] # Append new_number to existing
    return t # t stores list of umbers representing m long words of text


def find_matches_arithmetize(pattern,text,basis=2**16):
     """ find all occurrences of pattern in text
     using efficient arithmetization of text """
     assert(len(pattern) <= len(text))
     p=arithmetize(pattern,basis)
     t=arithmetize_text2(text,len(pattern),basis)
     matches=[]
     for s in range(len(t)):
         if p==t[s]: matches=matches+[s]
     return matches


def fingerprint(text,basis=2**16,r=2**32-3):
    """ used to compute karp-rabin fingerprint of the pattern
    employs Horner method (modulo r) """
    partial_sum=0
    for ch in text:
        partial_sum=(partial_sum*basis+ord(ch)) % r
    return partial_sum


def text_fingerprint(text,m,basis=2**16,r=2**32-3):
    """ used to computes karp-rabin fingerprint of the text """
    f=[]
    b_power=pow(basis,m-1,r)
    list.append(f, fingerprint(text[0:m], basis, r))
    # f[0] equals first text fingerprint
    for s in range(1,len(text)-m+1):
        new_fingerprint=((f[s-1]-ord(text[s-1])*b_power)*basis
        +ord(text[s+m-1])) % r
        # compute f[s], based on f[s-1]
        list.append(f,new_fingerprint)# append f[s] to existing f
    return f


def find_matches_KR(pattern,text,basis=2**16,r=2**32-3):
    """ find all occurrences of pattern in text
    using coin flipping Karp-Rabin algorithm """

    if len(pattern) > len(text):
        return []
    p = fingerprint(pattern,basis,r)
    f = text_fingerprint(text,len(pattern),basis,r)
    matches = [s for s, f_s in enumerate(f) if f_s == p]
    # list comprehension
    return matches


def sanitize(text):
    """Removes irrelevant features such as spaces and commas.
    :param text: A string of (index, character) tuples.
    """

    import re

    # NOTE: \p{L} or \p{Letter}: any kind of letter from any language.
    # http://www.regular-expressions.info/unicode.html
    p = re.compile(r'\w', re.UNICODE)

    def f(c):
        return p.match(c[1]) is not None

    return filter(f, map(lambda x: (x[0], x[1].lower()), text))

def kgrams(text, k=5):
    """Derives k-grams from text."""

    text = list(text)
    n = len(text)

    if n < k:
        yield text
    else:
        for i in range(n - k + 1):
            yield text[i:i+k]


def winnowing_hash(kgram):
    """
    :param kgram: e.g., [(0, 'a'), (2, 'd'), (3, 'o'), (5, 'r'), (6, 'u')]
    """
    kgram = zip(*kgram)
    kgram = list(kgram)

    # FIXME: What should we do when kgram is shorter than k?
    text = ''.join(kgram[1]) if len(kgram) > 1 else ''

    hash_function = default_hash
    hs = hash_function(text)

    # FIXME: What should we do when kgram is shorter than k?
    return (kgram[0][0] if len(kgram) > 1 else -1, hs)


def default_hash(text):
    import hashlib

    hs = hashlib.sha1(text.encode('utf-8'))
    hs = hs.hexdigest()[-4:]
    hs = int(hs, 16)

    return hs


def select_min(window):
    """In each window select the minimum hash value. If there is more than one
    hash with the minimum value, select the rightmost occurrence. Now save all
    selected hashes as the fingerprints of the document.
    :param window: A list of (index, hash) tuples.
    """

    #print window, min(window, key=lambda x: x[1])

    return min(window, key=lambda x: x[1])


def winnow_f(text, k=5):
    n = len(list(text))

    text = zip(range(n), text)
    text = sanitize(text)

    hashes = map(lambda x: winnowing_hash(x), kgrams(text, k))

    windows = kgrams(hashes, 4)

    return set(map(select_min, windows))


def get_shingles(s, k=5):
    """
    Shingle and discard the last k as there are just the last n<k characters from the document

    Parameters
    ----------

    s : string
       Document to shingle

    k : int
       Shingle length

    Returns
    -------

    list
        A list with all the shingles in the document

    """

    shingles = [s[i:i+k] for i in range(len(s))][:-5]
    return shingles

def jaccard_distance(set_a, set_b):
    """
    Get the Jaccard distance between two sets [ size of set intersection divided by the size of set union ]

    Parameters
    ----------

    set_a : set|list
        A set or list to compare

    set_b : set|list
        A set or list to compare to

    Returns
    -------

    floag
        The value of the Jaccard distance between these two sets

    """
    if type(set_a) or type(set_b) == type([]):
        set_a = set(set_a)
        set_b = set(set_b)
    size_intersection = len(set_a & set_b)
    size_union = len(set_a | set_b)
    try:
        jaccard_distance = size_intersection / size_union
    except:
        jaccard_distance = 0
    return jaccard_distance




