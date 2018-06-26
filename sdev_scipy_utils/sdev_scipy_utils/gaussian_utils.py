
from scipy.special import gammaln, multigammaln
import numpy as np


def gaussian_obs_log_likelihood(data, t, s):
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
