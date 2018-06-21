from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import pandas as pd
import numpy as np

def get_ngrams(text, n, flag='nltk' ):
    n_grams = ngrams(word_tokenize(text), n)
    if flag=='gensim':
        try:
            return np.array([ '_'.join(grams) for grams in n_grams])
        except Exception as e:
            print(e)
            return []
    elif flag=='nltk':
        try:
            return np.array([ ' '.join(grams) for grams in n_grams])
        except Exception as e:
            print(e)
            return []

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

def rank_ngrams(df_series, n):
    return df_series.apply(pd.Series).stack().groupby(level=0).apply(lambda x: x.drop_duplicates()).value_counts()[:n]

def get_max_len(series):
    length= 0
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
    tmp= []
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


def line_to_tensor(line, length = None):
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
    result =  tensor.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])
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
        result = np.concatenate((word[0], np.zeros(100 * (tensor_length - length) )))
        result = result.reshape(1, result.shape[0])
        return result
    return word


