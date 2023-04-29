
import numpy as np
from scipy.special import gammaln, multigammaln


def gaussian_obs_log_likelihood(data, t, s):
    """
        * type-def ::[Array] ::Int ::Int -> float
    * ---------------{Function}---------------
        * Computes the Gaussian observation log likelihood of the input data.
    * ----------------{Returns}---------------
        * -> log_likelihood ::float | The Gaussian observation log likelihood of the input data
    * ----------------{Params}----------------
        * : data ::Array[float] | The input data array
        * : t    ::int | The starting index of the data segment
        * : s    ::int | The ending index of the data segment
    * ----------------{Usage}-----------------
        * >>> gaussian_obs_log_likelihood(data, t, s)
        * -124.35817293583782
    * ----------------{Notes}-----------------
        * This function is used for calculating the Gaussian observation log likelihood of a segment of data.
        * The log likelihood can be useful in statistical modeling, such as for model selection or hypothesis testing.
    """
    s += 1
    n = s - t
    mean = data[t:s].sum(0) / n

    muT = (n * mean) / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = (
        1 + 0.5 * ((data[t:s] - mean) ** 2).sum(0) + ((n) / (1 + n)) * (mean ** 2 / 2)
    )
    scale = (betaT * (nuT + 1)) / (alphaT * nuT)

    # splitting the PDF of the student distribution up is /much/ faster.
    # (~ factor 20) using sum over for loop is even more worthwhile
    prob = np.sum(np.log(1 + (data[t:s] - muT) ** 2 / (nuT * scale)))
    lgA = (
        gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT / 2)
    )

    return np.sum(n * lgA - (nuT + 1) / 2 * prob)


def addNoise(xs, std=0.05):
    """
        * type-def ::[Array] ::float -> Array
    * ---------------{Function}---------------
        * Adds Gaussian noise to the input data array.
    * ----------------{Returns}---------------
        * -> noisy_data ::Array[float] | The input data array with added Gaussian noise
    * ----------------{Params}----------------
        * : xs  ::Array[float] | The input data array
        * : std ::float | The standard deviation of the Gaussian noise (default: 0.05)
    * ----------------{Usage}-----------------
        * >>> addNoise(xs, std)
        * array([0.95052724, 0.05234015, 0.99578864, ...])
    * ----------------{Notes}-----------------
        * The function adds Gaussian noise with a mean of 0 and the specified standard deviation.
        * This can be useful for simulating measurement errors or other random fluctuations in data.
    """
    return xs + np.random.normal(0, std, xs.shape)
