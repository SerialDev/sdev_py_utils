import pandas as pd
import numpy as np

def create_vector(x0, y0, z0, alpha, beta, gamma):
    # Create a vector with the given points and angles in each direction
    point = np.array([[x0], [y0], [z0]])
    alpha = np.cos(radians(alpha))
    beta = np.cos(radians(beta))
    gamma = np.cos(radians(gamma))
    # Cosines vector.
    cosin = np.array([[alpha], [beta], [gamma]])
    return np.array([point.reshape(3, 1), cosin.reshape(3, 1)])


def vector_generator(df):
    rw = df.iterrows()
    for i in enumerate(df):
        temp = rw.__next__()[1]
        x0, y0, z0 = temp[['PosX_attacker', 'PosY_attacker', 'PosZ_attacker']]
        alpha, beta, gamma = temp[['RotX_attacker', 'RotY_attacker', 'RotZ_attacker']]
        yield x0, y0, z0, alpha, beta, gamma


def df_vector_generator(df):
    vector = vector_generator(df)
    for i in enumerate(df):
        temp = vector.__next__()
        generated_vec = create_vector(temp[0], temp[1], temp[2],
                                      temp[3], temp[4], temp[5])
        yield generated_vec

        
def extract_log_file(csv_location):
    data = pd.read_csv(
        csv_location, 
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', 
        engine='python', 
        na_values='-', 
        header=None,
        usecols=[0, 3, 4, 5, 6, 7, 8],
        names=['ip', 'time', 'request', 'status', 'size', 'referer', 'user_agent'],
        converters={'time': parse_datetime,
                    'request': parse_str,
                    'status': int,
                    'size': int,
                    'referer': parse_str,
                    'user_agent': parse_str})
    return data


from utilities.decorators import dynamic_fn, countcalls
from utilities import decorators
from scipy.special import gammaln, multigammaln
@dynamic_fn
def gaussian_obs_log_likelihood(data, t, s):
    s += 1
    n = s - t
    mean = data[t:s].sum(0) / n

    muT = (n * mean) / (1 + n)
    nuT = 1 + n
    alphaT = 1 + n / 2
    betaT = 1 + 0.5 * ((data[t:s] - mean) ** 2).sum(0) + ((n)/(1 + n)) * (mean**2 / 2)
    scale = (betaT*(nuT + 1))/(alphaT * nuT)

    # splitting the PDF of the student distribution up is /much/ faster.
    # (~ factor 20) using sum over for loop is even more worthwhile
    prob = np.sum(np.log(1 + (data[t:s] - muT)**2/(nuT * scale)))
    lgA = gammaln((nuT + 1) / 2) - np.log(np.sqrt(np.pi * nuT * scale)) - gammaln(nuT/2)

    return np.sum(n * lgA - (nuT + 1)/2 * prob)




def root_mean_sqr(x, flag='norm'):
    if flag=='norm':
        return np.sqrt(x.dot(x)/x.size)
    if flag=='complex':
        return np.sqrt(np.vdot(x, x)/x.size)

def to_sarvam(data, save_path='', save_file=False):
    with open("tmp.plk", 'wb') as tmp:    
        pickle.dump(data, tmp)
        
        path = tmp.name
        ln = os.path.getsize(path)
        
    width = 128
        
    rem = ln % width
        
    a = array.array("B")
    
    with open("tmp.plk", 'rb') as tmp:
        a.fromfile(tmp, ln-rem)
    
    g = np.reshape(a, (len(a)//width, width))
    g = np.uint8(g)
    if save_file:
        file_format = 'png'
        file_path = save_path
        file_flag = os.path.exists('{}.{}'.format(file_path, file_format))
        #if file_flag:
        #    file_path = unique_filename_int(file_path)
        #scipy.misc.imsave(r'{}\\.{}'.format(file_path, file_format), g)
        scipy.misc.imsave(save_path, g)
    else:
        return g        
            



def get_compressed_file(data):
    """
    * Function: compress data and get the compressed and uncompressed size
    * Usage: get_compressed_file(data) . . . 
    * -------------------------------
    * This function returns 
    * size_compressed, size_uncompressed
    """
    with bz2.BZ2File('tmp.bzip', 'wb') as tmp:
        pickle.dump(data, tmp)
    compressed = open('tmp.bzip')
    size_compressed = os.path.getsize('tmp.bzip')
    with open('tmp.plk', 'wb') as tmp:
        pickle.dump(data, tmp)
    size_uncompressed = os.path.getsize('tmp.plk')
    return size_compressed, size_uncompressed

def log_df_cleaning(data):
    # Get Resource URI
    request = data.pop('request').str.split()
    # Select URL string
    data['resource'] = request.str[1]
    #filter out non-200 status codes
    data = data[(request.str[0] == 'GET') & (data.pop('status') == 200)]
    #filter robot.txt resources
    data = data[~data['resource'].str.match(
        r'^/media|^/static|^/admin|^/robots.txt$|^/favicon.ico$')]
    #filter out webcrawlers (Google, Yahoo and Bing) through the UserAgent
    data = data[~data['user_agent'].str.match(
        r'.*?bot|.*?spider|.*?crawler|.*?slurp', flags=re.I).fillna(False)]
    #filter out undesirable IPs from the webcrawlers
    data = data[~data['ip'].str.startswith('123.125.71.')]  # Baidu IPs.
    return data


def acm_booking_analysis(data):
    # obtain resource information
    resources = data['resource'].dropna()
    # Regex searchresults
    search_result_list = []
    for line in resources:
        line = line.rstrip()
        if re.search(r'/searchresults', line) :
            search_result_list.append(line)
    #Set cover problem, greedy approach
    accommodation_result_list = []
    for line in search_result_list:
        line = line.rstrip()
        if re.search(r'/accommodation', line) :
            accommodation_result_list.append(line) 
    # Append regex data into dataframe
    acm_results_df = pd.DataFrame(accommodation_result_list)
    acm_results_df.columns = ['accommodation']
    # Regex dataframe for page changes and store in another dataframe
    acm_next_df = acm_results_df[acm_results_df['accommodation'].str.contains("&p=[Z0-9]")]
    ## Percentage that go on to the next page
    acm_next_df.count()*100/acm_results_df.count()
    # Create a new dataframe to store the results
    counts = pd.DataFrame(acm_results_df.count(), columns=['made_search'])
    counts2 = pd.DataFrame(acm_next_df.count(), columns=['next_page'])
    frames = [counts, counts2]
    result = pd.concat(frames)
    return result


import os
def absoluteFilePaths(directory):
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


def dash2nan(x):
    if x == '-':
        x = np.nan
    else:
        x = float(x)/1048576.
    
    return x


def merge_date(df, year='Year', month='Month', day='Day'):
    """
    * Function: merge_date 
    * Usage: merge_date(DataFrame, col_year, col_month, col_day) . . .
    * -------------------------------
    * This function returns Datetime in the format YYYY-MM-DD from
    * input of a dataframe with columns holding 'Year', 'Month', 'Day' information
    * defaults to 'Year', 'Month', 'Day' if not specified
    * adds columns for Day and Month with default value of 1 if not present
    """
    if day not in df.columns:
        df['Day'] = 1
        
    if month not in df.columns:
        df['Month'] = 1
    
    df['DateTime'] = df[[year, month, day]].apply(lambda s : datetime.datetime(*s),axis = 1)
    return df


def clean_date(df, date='DateTime', year='Year', month='Month', day='Day'):
    """
    * Function: clean_date 
    * Usage: clean_date(Dataframe) . . .
    * -------------------------------
    * This function returns a DataFrame
    * with columns Year, Month, Day removed
    * if column DateTime is present
    """
    if date in df.columns:
        if year in df.columns:
            df.drop(year, axis=1, inplace=True)
        if month in df.columns:
            df.drop(month, axis=1, inplace=True)
        if day in df.columns:
            df.drop(day, axis=1, inplace=True)
    return df

def combine64(years, months=1, days=1, weeks=None, hours=None, minutes=None,
              seconds=None, milliseconds=None, microseconds=None, nanoseconds=None):
    """
    * Function: combine64 
    * Usage: df['combined']= combine64(df['Year'], df['Month'], df['Day']) . . .
    * -------------------------------
    * This function returns Datetime in the format YYYY-MM-DD:HH:mm:ss:ms:µs:ns
    """
    years = np.asarray(years) - 1970
    months = np.asarray(months) - 1
    days = np.asarray(days) - 1
    types = ('<M8[Y]', '<m8[M]', '<m8[D]', '<m8[W]', '<m8[h]',
             '<m8[m]', '<m8[s]', '<m8[ms]', '<m8[us]', '<m8[ns]')
    vals = (years, months, days, weeks, hours, minutes, seconds,
            milliseconds, microseconds, nanoseconds)
    return sum(np.asarray(v, dtype=t) for t, v in zip(types, vals)
               if v is not None)

def extract_log_file(csv_location):
    data = pd.read_csv(
        csv_location, 
        sep=r'\s(?=(?:[^"]*"[^"]*")*[^"]*$)(?![^\[]*\])', 
        engine='python', 
        na_values='-', 
        header=None,
        usecols=[0, 3, 4, 5, 6, 7, 8],
        names=['ip', 'time', 'request', 'status', 'size', 'referer', 'user_agent'],
        converters={'time': parse_datetime,
                    'request': parse_str,
                    'status': int,
                    'size': int,
                    'referer': parse_str,
                    'user_agent': parse_str})
    return data

from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    """
    * Function: test_stationarity
    * Usage: test_stationarity(Dataframe.timeseries) . . .
    * -------------------------------
    * This function returns plotted and detailed values for
    * stationarity testing of time series data
    """
    #Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries.unstack(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)


def tsplot(y, lags=None, figsize=(10, 8)):
    """
    * Function: tsplot
    * Usage: tsplot(ts_log_decompose, lags=10) . . .
    * -------------------------------
    * This function returns plotted timeseries data
    """
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    sm.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    sm.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(1.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

# Utility method: Given a dictionary and a threshold parameter K , the TOP K keys are returned according to element values
def get_top_keys(dictionary, top):
    items = dictionary.items()
    items.sorted(reverse=True, key=lambda x: x[1])
    return map(lambda x: x[0], items[:top])

def remove_outliers(group):
    """
    * Function: remove_outliers
    * Usage: remove_outliers(pandas.Series) . . .
    * -------------------------------
    * This function returns a pandas.Series (df['count'])
    * with the values over 3 Standard Deviations removed
    """
    mean, std = group.mean(), group.std()
    outliers = (group - mean).abs() > 3*std
    group[outliers] = mean        # or "group[~outliers].mean()"
    return group


def distance_cost_plot(distances):
    """
    * Function: distance_cost_plot
    * Usage: distance_cost_plot(np.matrix(distances)) . . .
    * -------------------------------
    * This function returns distance cost plot
    """
    im = plt.imshow(distances, interpolation='nearest', cmap='Reds') 
    plt.gca().invert_yaxis()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()
    plt.colorbar();

def path_cost(x, y, accumulated_cost, distances):
    """
    * Function: path_cost
    * Usage: path_cost((x, y, accumulated_cost, np.matrix(distances)) . . .
    * -------------------------------
    * This function returns path cost of a distance matrix
    """
    path = [[len(x)-1, len(y)-1]]
    cost = 0
    i = len(y)-1
    j = len(x)-1
    while i>0 and j>0:
        if i==0:
            j = j - 1
        elif j==0:
            i = i - 1
        else:
            if accumulated_cost[i-1, j] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                i = i - 1
            elif accumulated_cost[i, j-1] == min(accumulated_cost[i-1, j-1], accumulated_cost[i-1, j], accumulated_cost[i, j-1]):
                j = j-1
            else:
                i = i - 1
                j= j- 1
        path.append([j, i])
    path.append([0,0])
    for [y, x] in path:
        cost = cost +distances[x, y]
    return path, cost

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

import math

def average(x):
    """
    * Function: average
    * Usage: average(x) . . .
    * -------------------------------
    * This function returns average
    """
    assert len(x) > 0
    return float(sum(x)) / len(x)

import math
def pearson_def(x, y):
    """
    * Function: pearson_def
    * Usage: pearson_def(x, y) . . .
    * -------------------------------
    * This function returns pearson coefficient for two values
    """
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)


import math
 
def dynamicTimeWarp(seqA, seqB, d = lambda x,y: abs(x-y)):
    """
    * Function: dynamicTimeWarp
    * Usage: dynamicTimeWarp(dw['Count'], dw2['Count']) . . .
    * -------------------------------
    * This function returns Dynamic time warp between two numerical values
    """
    # create the cost matrix
    numRows, numCols = len(seqA), len(seqB)
    cost = [[0 for _ in range(numCols)] for _ in range(numRows)]
 
    # initialize the first row and column
    cost[0][0] = d(seqA[0], seqB[0])
    for i in range(1, numRows):
        cost[i][0] = cost[i-1][0] + d(seqA[i], seqB[0])
 
    for j in range(1, numCols):
        cost[0][j] = cost[0][j-1] + d(seqA[0], seqB[j])
 
    # fill in the rest of the matrix
    for i in range(1, numRows):
        for j in range(1, numCols):
            choices = cost[i-1][j], cost[i][j-1], cost[i-1][j-1]
            cost[i][j] = min(choices) + d(seqA[i], seqB[j])
 
    #for row in cost:
       # for entry in row:
            #print ("%03d" % entry,)
        #print ("")
    return cost[-1][-1]

def dtw_df(df, df2):
    """ Takes two dataframes and computes the dynamic time warp value
    for every pair of numbers into a new column DTW.
    Count is required as a Column name in order to use vectorized functions"""
    df['Count_two'] = 0
    df['DTW'] = 0
    for idx in range(len(df)):
        dtw = dynamicTimeWarp(list(df.iloc[[idx]]['Count']), list(df2.iloc[[idx]]['Count']))

        dw.Count_two.iloc[idx] = dw2.iloc[idx]['Count']
        df.DTW.iloc[idx] = dtw
        
    return df



def compute_cost(x, y, theta):
    inner = np.power(((x * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(x))

import numpy as np
def n_log_np(x, base=1):
    """ Considering the logarithm base change rule 
    give a numpy array and a base to get base n log"""
    return np.log(base**np.array(x))/np.log(base)


#!/usr/bin/python
#
# Stolen from Ero Carrera
# http://blog.dkbza.org/2007/05/scanning-data-for-entropy-anomalies.html

import math, string, sys, fileinput

def range_bytes (): return range(256)
def range_printable(): return (ord(c) for c in string.printable)
def H(data, iterator=range_bytes):
    if not data:
        return 0
    entropy = 0
    for x in iterator():
        p_x = float(data.count(chr(x)))/len(data)
        if p_x > 0:
            entropy += - p_x*math.log(p_x, 2)
    return entropy

def main ():
    for row in fileinput.input():
        string = row.rstrip('\n')
        print ("%s: %f" % (string, H(string, range_printable)))

for str in ['gargleblaster', 'tripleee', 'magnus', 'lkjasdlk',
               'aaaaaaaa', 'sadfasdfasdf', '7&wS/p(']:
    print ("%s: %f" % (str, H(str, range_printable)))


def entropy2(labels):
 """ Computes entropy of label distribution. """
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    counts = np.bincount(labels)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute standard entropy.
    for i in probs:
        ent -= i * log(i, base=n_classes)

    return ent


import math
from collections import Counter

def eta(data, unit='natural'):
    """ Compute Entropy """
    base = {
        'shannon' : 2.,
        'natural' : math.exp(1),
        'hartley' : 10.
    }

    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    probs = [float(c) / len(data) for c in counts.values()]
    probs = [p for p in probs if p > 0.]

    ent = 0

    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])

    return ent

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')

y, x = np.histogram(data , bins=200, normed=True)
x = (x + np.roll(x, -1))[:-1] / 2.0
# Distributions to check
DISTRIBUTIONS = [        
    st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
    st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
]
# best holders
best_distributions = st.norm
best_params = (0.0, 1.0)
best_sse = np.inf
# Create models from data
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, normed=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)
def make_pdf(dist, params, size=10000):
    """Generate distributions's Propbability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def plot_distributions(data, bins=50, col=''):
    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=bins, normed=True, alpha=0.5, color=plt.rcParams['axes.color_cycle'][1])
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'{} .\n All Fitted Distributions'.format(col))
    ax.set_xlabel(u'Temp (°C)')
    ax.set_ylabel('Frequency')

    # Find best fit distribution
    best_fit_name, best_fir_paramms = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Make PDF
    pdf = make_pdf(best_dist, best_fir_paramms)
    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'{} with best fit distribution'.format(col))
    ax.set_xlabel(u'X')
    ax.set_ylabel('Frequency')

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, normed=True, alpha=0.5, label='Data', legend=True, ax=ax)
    
    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fir_paramms)])
    dist_str = '{}({})'.format(best_fit_name, param_str)
    
    ax.set_title(u'{} with best fit distribution \n'.format(col) + dist_str)
    ax.set_xlabel(u'X')
    ax.set_ylabel('Frequency')



