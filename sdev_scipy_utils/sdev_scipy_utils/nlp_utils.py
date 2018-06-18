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
