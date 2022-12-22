"Put summary function here that prints or returns a dataframe"
import pandas as pd
from typing import Union
import empyrical
import numpy as np
from scipy.stats import skew, kurtosis, skewtest, kurtosistest

__all__ = [
    "sharpe_ratio",
    "calmar_ratio",
    "omega_ratio",
    "sortino_ratio",
    "tail_ratio",
    "annualised_returns",
    "annualised_volatility",
    "conditional_value_at_risk",
    "value_at_risk",
    "calculate_skewness",
    "calculate_kurtosis",
    "is_stable",
    "maximum_drawdown",
    "cumulative_returns",
    "elton_gruber_covariance",
    "covariance_shrinkage",
    "risk_contribution",
]


from typing import Tuple

import numpy as np


def shrinkage(returns: np.array) -> Tuple[np.array, float, float]:
    """
    Ledoit & Wolf constant correlation unequal variance shrinkage estimator.
    Shrinks sample covariance matrix towards constant correlation unequal variance matrix.
    Ledoit & Wolf ("Honey, I shrunk the sample covariance matrix", Portfolio Management, 30(2004),
    110-119) optimal asymptotic shrinkage between 0 (sample covariance matrix) and 1 (constant
    sample average correlation unequal sample variance matrix).
    Paper:
    http://www.ledoit.net/honey.pdf
    Matlab code:
    https://www.econ.uzh.ch/dam/jcr:ffffffff-935a-b0d6-ffff-ffffde5e2d4e/covCor.m.zip
    Special thanks to Evgeny Pogrebnyak https://github.com/epogrebnyak
    :param returns:
        t, n - returns of t observations of n shares.
    :return:
        Covariance matrix, sample average correlation, shrinkage.
    """
    t, n = returns.shape
    mean_returns = np.mean(returns, axis=0, keepdims=True)
    returns -= mean_returns
    sample_cov = returns.transpose() @ returns / t

    # sample average correlation
    var = np.diag(sample_cov).reshape(-1, 1)
    sqrt_var = var ** 0.5
    unit_cor_var = sqrt_var * sqrt_var.transpose()
    average_cor = ((sample_cov / unit_cor_var).sum() - n) / n / (n - 1)
    prior = average_cor * unit_cor_var
    np.fill_diagonal(prior, var)

    # pi-hat
    y = returns ** 2
    phi_mat = (y.transpose() @ y) / t - sample_cov ** 2
    phi = phi_mat.sum()

    # rho-hat
    theta_mat = ((returns ** 3).transpose() @ returns) / t - var * sample_cov
    np.fill_diagonal(theta_mat, 0)
    rho = (
        np.diag(phi_mat).sum()
        + average_cor * (1 / sqrt_var @ sqrt_var.transpose() * theta_mat).sum()
    )

    # gamma-hat
    gamma = np.linalg.norm(sample_cov - prior, "fro") ** 2

    # shrinkage constant
    kappa = (phi - rho) / gamma
    shrink = max(0, min(1, kappa / t))

    # estimator
    sigma = shrink * prior + (1 - shrink) * sample_cov

    return sigma, average_cor, shrink


def sharpe_ratio(
    price: Union[pd.DataFrame, pd.Series],
    riskFreeRate: float = 0.0,
    periodsPerYear: Union[float, int] = 252,
) -> float:
    """Calculates annualised sharpe ratio for given set of prices and risk free rate
    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
        historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising
    Returns
    -------
    float
        annualised sharpe ratio
    """
    returns = price.pct_change().dropna()
    rfPerPeriod = (1 + riskFreeRate) ** (1 / periodsPerYear) - 1
    excessReturn = returns - rfPerPeriod

    annualiseExcessReturn = annualised_returns(excessReturn, periodsPerYear)
    annualiseVol = annualised_volatility(returns, periodsPerYear)

    return annualiseExcessReturn / annualiseVol


def calmar_ratio(
    price: Union[pd.DataFrame, pd.Series],
    periodsPerYear: Union[float, int] = 252,
    riskFreeRate: float = 0.0,
) -> float:
    """Calculates annualised calmar ratio for given set of prices and risk free rate
    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
        historical prices of a given security
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising
    Returns
    -------
    float
        annualised calmar ratio
    """

    returns = price.pct_change().dropna()
    rfPerPeriod = (1 + riskFreeRate) ** (1 / periodsPerYear) - 1
    excessReturn = returns - rfPerPeriod

    annualiseExcessReturn = annualised_returns(excessReturn, periodsPerYear)
    calmar = annualiseExcessReturn / abs(maximum_drawdown(price))

    return calmar


def omega_ratio(
    price: Union[pd.DataFrame, pd.Series],
    riskFreeRate: float = 0.0,
    periodsPerYear: Union[float, int] = 252,
) -> float:
    """Calculates annualised omega ratio for given set of prices and risk free rate
    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
       historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    periodsPerYear : Union[float, int]
         periodicity of the returns data for purposes of annualising
    Returns
    -------
    float
        annualised omega ratio
    """
    if isinstance(price, pd.DataFrame):

        return price.apply(omega_ratio, axis=0)

    returns = price.pct_change().dropna()
    omega = empyrical.stats.omega_ratio(
        returns, risk_free=riskFreeRate, annualization=periodsPerYear
    )

    return omega


def sortino_ratio(
    price: Union[pd.DataFrame, pd.Series],
    periodsPerYear: Union[float, int] = 252,
    reqReturn: float = 0,
) -> float:
    """Calculates annualised sortino ratio for given set of prices and risk free rate
    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
       historical prices of a given security
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising
    reqReturn : float, optional
        the minimum acceptable return by investors, by default 0
    Returns
    -------
    float
        annualised sortino ratio
    """

    if isinstance(price, pd.DataFrame):

        return price.apply(sortino_ratio, axis=0)

    returns = price.pct_change().dropna()
    sortino = empyrical.stats.sortino_ratio(
        returns, annualization=periodsPerYear, required_return=reqReturn
    )

    return sortino


def tail_ratio(price: Union[pd.DataFrame, pd.Series]) -> float:
    """Calculates annualised tail ratio for given set of prices and risk free rate
    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
       historical prices of a given security
    Returns
    -------
    float
        annualised tail ratio
    """
    if isinstance(price, pd.DataFrame):

        return price.apply(tail_ratio, axis=0)

    returns = price.pct_change().dropna()
    tail = empyrical.stats.tail_ratio(returns)

    return tail


def annualised_returns(returns: pd.DataFrame, periodsPerYear: int = 252):
    """This function returns the annualised returns of a given dataframe of returns.
    If the freq of the data is not daily, the annualisation factor must be specified.
    The function returns nan if the value computed is too small
    Parameters
    ----------
    returns : pd.DataFrame
        dataframe of returns
    periodsPerYear : int, optional
        freq of returns in a year, by default 252
    Returns
    -------
    Annualised Returns
        Returns the annualised return for each column in the dataframe
    """

    compoundGrowth = (1 + returns).prod()
    nobs = returns.shape[0]
    return compoundGrowth ** (periodsPerYear / nobs) - 1


def annualised_volatility(returns: pd.DataFrame, periodsPerYear: int = 252):
    """This function returns the annualised volatility of a given dataframe of returns.
    If the freq of the data is not daily, the annualisation factor must be specified
    Parameters
    ----------
    returns : pd.DataFrame
        dataframe of returns
    periodsPerYear : int, optional
        freq of returns in a year, by default 252
    Returns
    -------
    Annualised Returns
        Returns the annualised volatility for each column in the dataframe
    """

    return returns.std() * (periodsPerYear ** 0.5)


def conditional_value_at_risk(price: pd.Series, threshold: float = 0.05) -> float:
    """Calculates Conditional Value at Risk for given price series
    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    Returns
    -------
    float
        Conditional Value at Risk (VaR value) for given price
    """

    returns = price.pct_change().dropna()

    cVar = np.mean(returns[returns < value_at_risk(price)])

    return cVar


def value_at_risk(price: pd.Series, threshold: float = 0.05) -> float:
    """Calculates Value at Risk for given price series
    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    Returns
    -------
    float
        Value at Risk (VaR value) for given price
    """

    import empyrical

    if isinstance(price, pd.DataFrame):

        return price.apply(value_at_risk)

    returns = price.pct_change().dropna()
    var = empyrical.stats.value_at_risk(returns, threshold)

    return var


def calculate_skewness(
    price: Union[pd.DataFrame, pd.Series], test: bool = False, **kwargs
) -> Union[float, pd.Series]:
    """Calculates the skewness for a given set of prices
    Parameters
    ----------
    price : Union[pd.DataFrame,pd.Series]
        historical prices of a given security
    Returns
    -------
    Union[float,pd.Series]
        skewness for a given set of prices
    """
    if test:
        result = skewtest(price, **kwargs)
        return result

    return skew(price)


def calculate_kurtosis(
    price: Union[pd.Series, pd.DataFrame], test: bool = False, **kwargs
) -> Union[float, pd.Series]:
    """Calculates the kurtosis for a given set of prices
    Parameters
    ----------
    price : Union[pd.DataFrame,pd.Series]
        historical prices of a given security
    Returns
    -------
    Union[float,pd.Series]
        kurtosis for a given set of prices
    """

    if test:

        result = kurtosistest(price, **kwargs)
        return result

    return kurtosis(price)


def is_stable(price: pd.Series) -> float:
    """Calculates stability for a given set of prices
    Parameters
    ----------
    price : pd.Series
       historical prices of a given security
    Returns
    -------
    float
       stability for a given set of prices
    """
    if isinstance(price, pd.DataFrame):
        return price.apply(is_stable)

    returns = price.pct_change().dropna()
    stability = empyrical.stats.stability_of_timeseries(returns)
    return stability


def maximum_drawdown(price: pd.Series) -> float:
    """Calculates maximum drawdown for a given set of prices
    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    Returns
    -------
    float
        maximum drawdown for a given set of prices
    """

    maximum_drawdown.drawdowns = (price / price.cummax()) - 1
    maximumDrawdown = (maximum_drawdown.drawdowns).min()

    return maximumDrawdown


def cumulative_returns(price: Union[pd.DataFrame, pd.Series]) -> float:
    """Calculates cumulative returns for a given set of prices
    Parameters
    ----------
    price : Union[pd.DataFrame, pd.Series]
        historical prices of a given security
    Returns
    -------
    float
        cumulative returns for a given set of prices
    """
    returns = price.pct_change().dropna()
    cumReturns = empyrical.stats.cum_returns(returns)

    return cumReturns


def alpha(
    price: pd.Series,
    marketReturn: float,
    riskFreeRate: float = 0.0,
    periodsPerYear: Union[float, int] = 252,
) -> float:
    """Calculates annualised alpha for a given set of prices, risk free rate and benchmark return (market return in CAPM)
    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    marketReturn : float
        daily noncumulative benchmark return throughout the period
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising
    Returns
    -------
    float
        annualised alpha for a given set of prices, risk free rate and benchmark return (market return in CAPM)
    """

    raise NotImplementedError("Will do it later :P")
    if isinstance(price, pd.DataFrame):
        return price.apply(alpha, args=(marketReturn, riskFreeRate, periodsPerYear))

    returns = price.pct_change().dropna()
    a = empyrical.stats.alpha(
        returns,
        factor_returns=marketReturn,
        risk_free=riskFreeRate,
        annualization=periodsPerYear,
    )

    return a


def beta(
    price: pd.Series,
    riskFreeRate: float,
    marketReturn: float,
    periodsPerYear: Union[float, int],
) -> float:
    """Calculates annualised beta for a given set of prices, risk free rate and benchmark return (market return in CAPM)
    Parameters
    ----------
    price : pd.Series
        historical prices of a given security
    riskFreeRate : float
        given constant risk free rate throughout the period
    marketReturn : float
        daily noncumulative benchmark return throughout the period
    periodsPerYear : Union[float, int]
        periodicity of the returns data for purposes of annualising
    Returns
    -------
    float
        annualised beta for a given set of prices, risk free rate and benchmark return (market return in CAPM)
    """
    # TO DO: take dataframe as price input
    raise NotImplementedError("Will do it later :P")
    r = price.pct_change().dropna()
    b = empyrical.stats.beta(
        r,
        factor_returns=marketReturn,
        risk_free=riskFreeRate,
        annualisation=periodsPerYear,
    )
    return b


def elton_gruber_covariance(price: pd.DataFrame, **kwargs):
    """This function estimates the covariance matrix by assuming an implicit structure as defined by the
    Elton-Gruber Constant Correlation model.
    Parameters
    ----------
    price : pd.DataFrame
        Historical prices of a given security
    Returns
    -------
    pd.DataFrame
        Returns a covariance matrix
    """

    returns = price.pct_change().dropna()
    rhos = returns.corr()
    n = rhos.shape[0]
    rhoBar = (rhos.values.sum() - n) / (n * (n - 1))
    constantCorrelation = np.full_like(rhos, rhoBar)
    np.fill_diagonal(constantCorrelation, 1.0)
    standardDev = returns.std()

    result = pd.DataFrame(
        constantCorrelation * np.outer(standardDev, standardDev),
        index=returns.columns,
        columns=returns.columns,
    )

    return result


def covariance_shrinkage(price: pd.DataFrame, delta: float = 0.5, **kwargs):
    """This function computes the covariance matrix using the Ledoit-Wolf covariance shrinkage method
    taking a linear combination of the Constant Correlation matrix, acting as our prior and the Sample covariance matrix. The posterior covariance matrix is then computed.
    Parameters
    ----------
    price : pd.DataFrame
        Historical prices of a given security
    delta : float, optional
        Constant by which to weigh the priori matrix, by default 0.5
    Returns
    -------
    pd.DataFrame
        Returns a covariance matrix
    """

    returns = price.pct_change().dropna()
    sampleCovariance = returns.cov()
    priorCovariance = elton_gruber_covariance(price, **kwargs)

    result = delta * priorCovariance + (1 - delta) * sampleCovariance

    return result


def risk_contribution(
    portfolioWeights: Union[np.array, pd.DataFrame], covarianceMatrix: pd.DataFrame
):
    """This function computes the contributions to the risk/variance of the constituents of a portfolio, given
    a set of portfolio weights and a covariance matrix
    Parameters
    ----------
    portfolioWeights : Union[np.array, pd.DataFrame]
        weights of our assets in our portfolio
    covarianceMatrix : pd.DataFrame
        the covariance matrix of our assets computed by any method
    Returns
    -------
    pd.DataFrame
        Returns the risk contribution of each asset
    """

    portfolioVariance = np.dot(
        np.dot(portfolioWeights.T, covarianceMatrix), portfolioWeights
    )
    marginalContribution = np.dot(covarianceMatrix, portfolioWeights)
    riskContribution = (
        np.multiply(marginalContribution, portfolioWeights) / portfolioVariance
    )

    return riskContribution
