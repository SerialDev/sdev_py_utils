"""Anomaly detection algos using the scipy stack"""

import numpy as np
from math import sqrt
import scipy
import scipy.stats

# -----------{Peak detection}-----------#


def detect_peaks(signal, threshold=0.5):
    """
    Performs peak detection on three steps:
    root mean square,
    peak to average ratios and first order logic.
    Threshold used to discard peaks too small

    Parameters
    ----------

    signal : np.array
        numpy array containing the signal to process

    threshold : float
        floating point value to assign a threshold

    Returns
    -------

    list
        A list of indexes where a peak was found
    """

    # compute root mean square
    root_mean_square = sqrt(np.sum(np.square(signal) / len(signal)))
    # compute peak to average ratios
    ratios = np.array([pow(x / root_mean_square, 2) for x in signal])
    # apply first order logic
    peaks = (
        (ratios > np.roll(ratios, 1))
        & (ratios > np.roll(ratios, -1))
        & (ratios > threshold)
    )
    # optional: return peak indices
    peak_indexes = []
    for i in range(0, len(peaks)):
        if peaks[i]:
            peak_indexes.append(i)
    return peak_indexes


def findpeaks(data, spacing=1, limit=None):
    """
    Find peaks in `data` qhich are of `spacing` width and >= `limit`

    Parameters
    ----------

    data : list
       A list of values to check for peaks

    spacing : uint
       Minimum spacing to the next peak(should be 1 or more)

    limit : None|uint
       Peaks should have a value greater or equal

    Returns
    -------

    list
        A list of peak indexes


    """
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.0e-6
    x[-spacing:] = data[-1] - 1.0e-6
    x[spacing : spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start : start + len]  # before
        start = spacing
        h_c = x[start : start + len]  # central
        start = spacing + s + 1
        h_a = x[start : start + len]  # after
        peak_candidate = np.logical_and(
            peak_candidate, np.logical_and(h_c > h_b, h_c > h_a)
        )

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


# ----------{Anomaly detection}---------#


def tail_avg(series, num_points):
    """
    This is a utility function used to calculate the average
    of the last `num_points` datapoints in the series as a
    measure, instead of just the last datapoint.
    It reduces noise, but it also reduces sensitivity
    and increases the delay to detection.

    Parameters
    ----------

    series : pd.Series
       A pandas series to get the tail avg from

    num_points : uint
       number of points to use as tail

    Returns
    -------

    float
        Tail average


    """
    t = series[-num_points:].sum() / num_points
    return t


def median_absolute_deviation(timeseries, threshold=6):
    """
    A timeseries is anomalous if the deviation
    of its latest datapoint with respect to
    the median is X times larger than the median
    of deviations

    Parameters
    ----------

    timeseries : pd.Series
       A pandas timeseries to check MAD from

    threshold : uint
       What threshold to use to define a anomaly

    Returns
    -------

    Tuple
        A tuple containing [0]:Bool, [1]:float, [2]:uint

    """
    series = timeseries
    try:
        median = series.median()
        demedianed = np.abs(series - median)
        median_deviation = demedianed.median()
        # The test statistic is infinite when the median is zero,
        # so it becomes super sensitive. We play it safe and skip when this happens.
        if median_deviation == 0:
            return False
        test_statistic = demedianed.iloc[-1] / median_deviation
        # Completely arbitary...triggers if the median deviation is
        # [threshold]6 times bigger than the median
        if test_statistic > threshold:
            return True, test_statistic, threshold
        else:
            return False, test_statistic, threshold

    except AttributeError:
        median = np.median(series)
        demedianed = np.abs(series - median)
        median_deviation = np.median(demedianed)
        # The test statistic is infinite when the median is zero,
        # so it becomes super sensitive. We play it safe and skip when this happens.
        if median_deviation == 0:
            return False
        test_statistic = demedianed[-1] / median_deviation
        # Completely arbitary...triggers if the median deviation is
        # [threshold]6 times bigger than the median
        if test_statistic > threshold:
            return True, test_statistic, threshold
        else:
            return False, test_statistic, threshold


def grubbs(series):
    """
    A timeseries is anomalous if the Z score is
    greater than the Grubb's score.

    Parameters
    ----------

    series : pd.Series
       A pandas series being checked for anomalies

    Returns
    -------

    Tuple
         [0]:Bool,[1]:float,[2]:float

    """

    # series = scipy.array([x[1] for x in timeseries])
    stdDev = scipy.std(series)
    mean = np.mean(series)
    tail_average = tail_avg(series, 30)
    z_score = (tail_average - mean) / stdDev
    len_series = len(series)
    threshold = scipy.stats.t.isf(0.05 / (2 * len_series), len_series - 2)
    threshold_squared = threshold * threshold
    grubbs_score = ((len_series - 1) / np.sqrt(len_series)) * np.sqrt(
        threshold_squared / (len_series - 2 + threshold_squared)
    )
    result = z_score > grubbs_score
    result = getattr(result, "tolist")()
    return (result, grubbs_score, z_score)


def stddev_from_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the
    average of the latest three datapoints - the moving average
    is greater than three standard deviations of the average.
    This does not exponentially weight the MA and so is better
    for detecting anomalies with respect to the entire series

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """
    series = timeseries
    mean = series.mean()
    stdDev = series.std()
    t = tail_avg(timeseries, 30)

    return abs(t - mean) > 3 * stdDev, stdDev


def stddev_from_moving_average(timeseries):
    """
    A timeseries is anomalous if the absolute value of the average of the latest three datapoint minus the moving average is greater than three standard deviations of the moving average. This is better for finding anomalies with respect to the short term trends

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """
    series = timeseries
    expAverage = timeseries.ewm(com=50).mean()
    stdDev = timeseries.ewm(com=50).std(bias=False)

    return abs(series.iloc[-1] - expAverage.iloc[-1]) > 3 * stdDev.iloc[-1]


def mean_subtraction_cumulation(timeseries):
    """
    A timeseries is anomalous if the value of the next datapoint in the
    series is farther than three standard deviations out in cumulative terms
    after subtracting the mean from each data point.

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """

    # series = pandas.Series([x[1] if x[1] else 0 for x in timeseries])
    series = timeseries
    series = series - series[0 : len(series) - 1].mean()
    stdDev = series[0 : len(series) - 1].std()
    expAverage = series.ewm(com=15).mean()

    return abs(series.iloc[-1]) > 3 * stdDev


def least_squares(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints
    on a projected least squares model is greater than three sigma.

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """

    # x = np.array([t[0] for t in timeseries])
    # y = np.array([t[1] for t in timeseries])
    x = timeseries
    y = timeseries
    A = np.vstack([x, np.ones(len(x))]).T
    results = np.linalg.lstsq(A, y)
    residual = results[1]
    m, c = np.linalg.lstsq(A, y)[0]
    errors = []
    for i, value in enumerate(y):
        projected = m * x.iloc[i] + c
        error = value - projected
        errors.append(error)

        if len(errors) < 3:
            return False

        std_dev = scipy.std(errors)
        t = (errors[-1] + errors[-2] + errors[-3]) / 3

        return abs(t) > std_dev * 3 and round(std_dev) != 0 and round(t) != 0


def histogram_bins(timeseries):
    """
    A timeseries is anomalous if the average of the last three datapoints falls
    into a histogram bin with less than 20 other datapoints (you'll need to tweak
    that number depending on your data)

    Returns: the size of the bin which contains the tail_avg. Smaller bin size
    means more anomalous.

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """

    # series = scipy.array([x[1] for x in timeseries])
    series = timeseries
    t = tail_avg(timeseries, 30)
    h = np.histogram(series, bins=15)
    bins = h[1]
    for index, bin_size in enumerate(h[0]):
        if bin_size <= 20:
            # Is it in the first bin?
            if index == 0:
                if t <= bins[0]:
                    return True
                # Is it in the current bin?
            elif t >= bins[index] and t < bins[index + 1]:
                return True
    return False


def ks_test(timeseries):
    """
    A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
    that data distribution for last 10 minutes is different from last hour.
    It produces false positives on non-stationary series so Augmented
    Dickey-Fuller test applied to check for stationarity.
    """

    # hour_ago = time() - 3600
    # ten_minutes_ago = time() - 600
    # reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
    # probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])
    reference = timeseries.iloc[-300:]
    probe = timeseries.iloc[-30:]
    if reference.size < 20 or probe.size < 20:
        return False

    ks_d, ks_p_value = scipy.stats.ks_2samp(reference, probe)

    if ks_p_value < 0.05 and ks_d > 0.5:
        adf = sm.tsa.stattools.adfuller(reference, 10)
        if adf[1] < 0.05:
            return True, ks_d, ks_p_value

    return False, ks_d, ks_p_value


def ks_test_ljung(timeseries):
    """
    A timeseries is anomalous if 2 sample Kolmogorov-Smirnov test indicates
    that data distribution for last 10 minutes is different from last hour.
    It produces false positives on series with trends so Ljung-Box test for
    no autocorrelation is used to filter them out.

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """

    # hour_ago = time() - 3600
    # ten_minutes_ago = time() - 600
    # reference = scipy.array([x[1] for x in timeseries if x[0] >= hour_ago and x[0] < ten_minutes_ago])
    # probe = scipy.array([x[1] for x in timeseries if x[0] >= ten_minutes_ago])
    reference = timeseries.iloc[-300:]
    probe = timeseries.iloc[-30:]

    if reference.size < 20 or probe.size < 20:
        return False

    ks_d, ks_p_value = scipy.stats.ks_2samp(reference, probe)

    if ks_p_value < 0.05 and ks_d > 0.5:
        _, ljp = sm.stats.diagnostic.acorr_ljungbox(reference)
        if ljp[-1] > 0.05:
            return True, ks_d, ks_p_value

    return False, ks_d, ks_p_value


def bayesian_anomaly(series):
    """
    * Function: bayesian anomaly detection
    * -----------{returns}------------
    * returns true if [-5:5] of argmax(anomaly) is > than stdev . . .
    * ------------{usage}-------------
    >>> bayesian_anomaly(pd.series)

    Parameters
    ----------

    timeseries : pd.Series
       Timeseries dataset to check for anomalies

    Returns
    -------

    Bool
         A Boolean value of whether it is anomalous or not
    """
    index = bayesian_changepoint(series)[1]
    MAD = series.mad()
    subset = series[index - 2 : index + 5].mean()
    return subset > MAD


# {Reconstruction error}#


# Establish a reconstruction error baseline (maybe MAD for rec error)
from sklearn import cluster


# def reconstruction_algo_anomaly(
#     data,
#     segment_length,
#     slide_length,
#     func=sin_window_func,
#     clusterer=cluster.KMeans,
#     flag="test",
#     plot=False,
# ):

#     """
#     * Function: Anomaly detection based on reconstruction of clustered data
#     * Usage: . . .
#     * function is a windowing function that takes in a segment and lenght func(segment, len)
#     * -------------------------------
#     * This function returns
#     * clusterer: cluster.[KMeans, AffinityPropagation, DBSCAN, SpectralClustering, etc..]
#     """
#     segments = sliding_window(data, segment_length, slide_length, flag="chunks")
#     windowed_segments = []
#     for segment in segments:
#         windowed_segments.append(inf_nan_tozero(func(segment, segment_length)))
#     clusterer = clusterer(n_clusters=10)
#     clusterer.fit(windowed_segments)
#     centroids = clusterer.cluster_centers_
#     if flag == "test":
#         windowed = windowed_segments[0]
#         nearest_centroid_idx = clusterer.predict(windowed)[0]
#         nearest_centroid = np.copy(centroids[nearest_centroid_idx])
#         if plot == True:
#             import matplotlib.pyplot as plt

#             plt.figure()
#             plt.plot(segments[0], label="Original Segment")
#             plt.plot(windowed, label="Windowed segment")
#             plt.plot(nearest_centroid, label="Nearest_centroid")
#             plt.legend()
#             plt.show()
#         return segments[0], windowed, nearest_centroid
#     else:
#         # return windowed_segments, clusterer
#         reconstruction = np.zeros(len(data))
#         n_plot_samples = 30
#         for segment_n, segment in enumerate(windowed_segments):
#             segment = np.copy(segment)
#             # segment = inf_nan_tozero(func(segment, 10))
#             nearest_centroid_idx = clusterer.predict(segment)[0]
#             centroids = clusterer.cluster_centers_
#             nearest_centroid = np.copy(centroids[nearest_centroid_idx])

#             pos = segment_n * slide_length
#             reconstruction[pos : pos + segment_length] += nearest_centroid
#         error = reconstruction[0:n_plot_samples] - data[0:n_plot_samples]
#         error_98th_percentile = np.percentile(error, 98)
#         if plot == True:
#             import matplotlib.pyplot as plt

#             plt.figure()
#             plt.plot(data[0:n_plot_samples], label="Original data")
#             plt.legend()
#             plt.show()
#             plt.plot(reconstruction[0:n_plot_samples], label="Reconstructed_data")
#             plt.legend()
#             plt.show()
#             plt.plot(error[0:n_plot_samples], label="Reconstruction Error")
#             plt.legend()
#             plt.show()
#             plt.plot(data[0:n_plot_samples], label="Original data")
#             plt.plot(reconstruction[0:n_plot_samples], label="Reconstructed_data")
#             plt.plot(error[0:n_plot_samples], label="Reconstruction Error")
#             plt.legend()
#             plt.show()
#             print("Maximum reconstruction error was {:0.1f}".format(error.max()))
#             print(
#                 "98th percentile of reconstruction error was {:0.1f} ".format(
#                     error_98th_percentile
#                 )
#             )
#         if error.max() > 30:
#             return True
#         else:
#             return False
#     return reconstruction, error, windowed_segments


# --------{Changepoint Detection}-------#


def bayesian_changepoint(data):
    """
    * ---------------{Function}---------------
    * Implements the Bayesian Changepoint Detection algorithm to detect changes in the underlying distribution of a sequence of data
    * ----------------{Returns}---------------
    * -> y ::List[float] |A list of posterior probabilities for each time step being a changepoint
    * -> zz ::int |The index of the changepoint with the highest posterior probability
    * -> mean1 ::float |The mean of the data up to the changepoint
    * -> mean2 ::float |The mean of the data after the changepoint
    * -> p ::List[int] |The indices of the top 5 changepoints with the highest posterior probabilities
    * ----------------{Params}----------------
    * : data ::List[float] |A list of numerical data
    * ----------------{Usage}-----------------
    * >>> bayesian_changepoint([1,2,3,4,5,6,7,8,9,10])
    * ([0.0, 0.4734455262454487, 6.634482216567264, 9.368698106198455, 8.421528117252147, 5.665903308613675, 2.4870415230581857, 0.8398807664147167, 0.14159425310363395],
    * 1,
    * 1.5,
    * 6.0,
    * [1, 2, 3, 4, 5])
    * ----------------{References}-----------------
    * https://projecteuclid.org/journals/annals-of-statistics/volume-23/issue-4/Estimation-of-Change-Point-Vectors-by-the-Bayesian-Information-Criterion/10.1214/aos/1176325766.full
    """
    n = len(data)
    # dbar = sum( data )/float(n)
    dbar = np.mean(data)

    # dsbar = sum (data*data)/float(n)
    dsbar = np.mean(np.multiply(data, data))

    fac = dsbar - np.square(dbar)

    summ = 0
    summup = []
    for z in range(n):
        summ += data[z]
        summup.append(summ)

    y = []

    for m in range(n - 1):
        pos = m + 1
        mscale = 4 * (pos) * (n - pos)
        Q = summup[m] - (summ - summup[m])
        U = -np.square(dbar * (n - 2 * pos) + Q) / float(mscale) + fac
        y.append(
            -(n / float(2) - 1) * np.log(n * U / 2) - 0.5 * np.log((pos * (n - pos)))
        )

    z, zz = np.max(y), np.argmax(y)
    mean1 = sum(data[: zz + 1] / float(len(data[: zz + 1])))
    mean2 = sum(data[(zz + 1) : n]) / float(n - 1 - zz)
    # p = y.argsort()[-3:][::-1]
    p = sorted(range(len(y)), key=lambda x: y[x])[-5:]
    return y, zz, mean1, mean2, p


# ------{Digital Signal Processing}-----#


def sliding_window(data, segment_length, slide_length, flag="chunks"):
    """
        * type-def ::[Array] ::Int ::Int ::str -> [List[Array]] | iterator
    * ---------------{Function}---------------
        * Generates sliding window segments of the input data.
    * ----------------{Returns}---------------
        * -> segments ::List[Array[float]] | A list of the sliding window segments
        * -> iterator ::iterator | An iterator over the sliding window segments
    * ----------------{Params}----------------
        * : data          ::Array[float] | The input data array
        * : segment_length ::int | The length of each segment
        * : slide_length  ::int | The step size between segments
        * : flag          ::str | "chunks" for a list of segments, "lazy" for an iterator
    * ----------------{Usage}-----------------
        * >>> sliding_window(data, segment_length, slide_length)
        * [array([segment_1_element_1, ...]), array([segment_2_element_1, ...]), ...]
    * ----------------{Notes}-----------------
        * The function has two modes, determined by the 'flag' parameter.
        * If 'flag' is set to "chunks", the function returns a list of all segments.
        * If 'flag' is set to "lazy", the function returns an iterator that generates segments on the fly.
    """

    def iter_sliding_window(data, segment_length, slide_length):
        for start_position in range(0, len(data), slide_length):
            end_position = start_position + segment_length
            yield data[start_position:end_position]

    def bulk_sliding_window(data, segment_length, slide_length):
        segments = []
        for start_position in range(0, len(data), slide_length):
            end_position = start_position + segment_length
            # make a copy so changes to 'segments doesn't modify original data
            segment = np.copy(data[start_position:end_position])
            # if we're at the end and we've got a truncated segment, drop it
            if len(segment) != segment_length:
                continue
            segments.append(segment)
        print("Produced {} waveform segments".format(len(segments)))
        return segments

    if flag == "chunks":
        return bulk_sliding_window(data, segment_length, slide_length)
    elif flag == "lazy":
        return iter_sliding_window(data, segment_length, slide_length)


def spectral_centroid(x, samplerate=44100):
    """
        * type-def ::[Array] ::Int -> float
    * ---------------{Function}---------------
        * Computes the spectral centroid of the input signal.
    * ----------------{Returns}---------------
        * -> centroid ::float | The spectral centroid of the input signal
    * ----------------{Params}----------------
        * : x          ::Array[float] | The input signal data array
        * : samplerate ::int | The sample rate of the input signal (default: 44100 Hz)
    * ----------------{Usage}-----------------
        * >>> spectral_centroid(x, samplerate)
        * 5321.784295378295
    * ----------------{Notes}-----------------
        * The spectral centroid is a measure used in digital signal processing to characterize the spectral content of a signal.
        * It represents the center of mass of the spectrum and can be used to differentiate sounds with different timbral qualities.
        * weighted mean of the frequencies present in the signal,
        * determined using a Fourier transform,
        * with their magnitudes as the weights
    """
    magnitudes = np.abs(np.fft.rfft(x))  # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(
        np.fft.fftfreq(length, 1.0 / samplerate)[: length // 2 + 1]
    )  # positive frequencies
    return np.sum(magnitudes * freqs) / np.sum(magnitudes)  # return weighted mean


def rectangular_window_func(data, segment_length):
    """
        * type-def ::[Array] ::Int -> [Array]'
    * ---------------{Function}---------------
        * Returns the input data unmodified (rectangular window).
    * ----------------{Returns}---------------
        * -> data ::Array[float] | The unmodified input data
    * ----------------{Params}----------------
        * : data          ::Array[float] | The input data array
        * : segment_length ::int | The length of the window (unused)
    * ----------------{Usage}-----------------
        * >>> rectangular_window_func(data, segment_length)
        * array([data_element_1, data_element_2, ...])
    """
    return data


def sin_window_func(data, segment_length):
    """
        * type-def ::[Array] ::Int -> [Array]'
    * ---------------{Function}---------------
        * Applies a sin^2 window function to the input data.
    * ----------------{Returns}---------------
        * -> windowed_segment ::Array[float] | The input data with the sin^2 window applied
    * ----------------{Params}----------------
        * : data          ::Array[float] | The input data array
        * : segment_length ::int | The length of the window
    * ----------------{Usage}-----------------
        * >>> sin_window_func(data, segment_length)
        * array([windowed_data_element_1, windowed_data_element_2, ...])
    """
    window_rads = np.linspace(0, np.pi, segment_length)
    window = np.sin(window_rads) ** 2
    windowed_segment = data * window
    return windowed_segment


def inf_nan_tozero(data):
    """
        * type-def ::[Array] -> [Array]'
    * ---------------{Function}---------------
        * Replaces infinite and NaN values in the input data with zeros.
    * ----------------{Returns}---------------
        * -> data ::Array[float] | The input data with infinite and NaN values replaced by zeros
    * ----------------{Params}----------------
        * : data       ::Array[float] | The input data array
    * ----------------{Usage}-----------------
        * >>> inf_nan_tozero(data)
        * array([data_element_1, data_element_2, ...])
    """
    data[data == -np.inf] = 0
    data[data == np.inf] = 0
    data = np.nan_to_num(data)
    return data


def log10_window_func(data, segment_length):
    """
        * type-def ::[Array] ::Int -> [Array]'
    * ---------------{Function}---------------
        * Applies a log10 window function to the input data.
    * ----------------{Returns}---------------
        * -> windowed_segment ::Array[float] | The input data with the log10 window applied
    * ----------------{Params}----------------
        * : data          ::Array[float] | The input data array
        * : segment_length ::int | The length of the window (unused)
    * ----------------{Usage}-----------------
        * >>> log10_window_func(data, segment_length)
        * array([windowed_data_element_1, windowed_data_element_2, ...])
    """
    window = np.log10(data)
    windowed_segment = data * window
    return inf_nan_tozero(windowed_segment)


def gradient_window_func(data, segment_length):
    """
    * ---------------Typedef----------------
    * type-def ::(array_like, int) -> array_like
    * ---------------Function---------------
    * This function calculates the gradient of the input data and returns the result as a window.
    *
    * ----------------Returns---------------
    * -> result ::array_like
    *   The gradient of the input data.
    * ----------------Params----------------
    * data ::array_like
    *   The input data to calculate the gradient of.
    * segment_length ::int
    *   Not used in the current implementation.
    * ----------------Usage-----------------
    * This function can be used to calculate the gradient of a signal or an array of values.
    * Example:
    * gradient_window_func([1, 2, 3, 4, 5], 2) -> [1., 1., 1., 1.]
    * ----------------Notes-----------------
    * The segment_length parameter is currently not used in the function implementation.
    * The function uses the numpy gradient function to calculate the gradient of the input data.

    """
    window = np.gradient(data, 1)
    return window


# ---------{Frequency detection}--------#

from scipy.signal import fftconvolve
from scipy.signal import kaiser


def find(condition):
    "Return the indices where ravel(condition) is true"
    (res,) = np.nonzero(np.ravel(condition))
    return res


def freq_from_crossings(signal, fs):
    """Estimate frequency by counting zero crossings

    Pros: Fast, accurate (increasing with signal length). Works well for long
    low-noise sines, square, triangle, etc.

    Cons: Doesn't work if there are multiple zero crossings per cycle,
    low-frequency baseline shift, noise, etc.

    """
    # Find all indices right before a rising-edge zero crossing
    indices = find((signal[1:] >= 0) & (signal[:-1] < 0))

    # Naive (Measures 1000.185 Hz for 1000 Hz, for instance)
    # crossings = indices

    # More accurate, using linear interpolation to find intersample
    # zero-crossings (Measures 1000.000129 Hz for 1000 Hz, for instance)
    crossings = [i - signal[i] / (signal[i + 1] - signal[i]) for i in indices]

    # Some other interpolation based on neighboring points might be better.
    # Spline, cubic, whatever

    return fs / np.mean(np.diff(crossings))


def parabolic(f, x):
    """
    Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
    f is a vector and x is an index for that vector.
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
    """
    xv = 1 / 2.0 * (f[x - 1] - f[x + 1]) / (f[x - 1] - 2 * f[x] + f[x + 1]) + x
    yv = f[x] - 1 / 4.0 * (f[x - 1] - f[x + 1]) * (xv - x)
    return (xv, yv)


def freq_from_fft(signal, fs):
    """Estimate frequency from peak of FFT

    Pros: Accurate, usually even more so than zero crossing counter
    (1000.000004 Hz for 1000 Hz, for instance). Due to parabolic
    interpolation being a very good fit for windowed log FFT peaks?
    https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html
    Accuracy also increases with signal length

    Cons: Doesn't find the right value if harmonics are stronger than
    fundamental, which is common.

    """

    N = len(signal)

    # Compute Fourier transform of windowed signal
    windowed = signal * kaiser(N, 100)
    f = np.fft.rfft(windowed)
    # Find the peak and interpolate to get a more accurate peak
    i_peak = np.argmax(abs(f))  # Just use this value for less-accurate result
    i_interp = parabolic(np.log(abs(f)), i_peak)[0]
    # Convert to equivalent frequency
    return fs * i_interp / N, i_interp  # N Hz


def freq_from_autocorr(signal, fs):
    """Estimate frequency using autocorrelation

    Pros: Best method for finding the true fundamental of any repeating wave,
    even with strong harmonics or completely missing fundamental

    Cons: Not as accurate, doesn't work for inharmonic things like musical
    instruments, this implementation has trouble with finding the true peak

    """
    # Calculate autocorrelation (same thing as convolution, but with one input
    # reversed in time), and throw away the negative lags
    signal -= np.mean(signal)  # Remove DC offset
    corr = fftconvolve(signal, signal[::-1], mode="full")
    corr = corr[len(corr) // 2 :]

    # Find the first low point
    d = np.diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag). This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    i_peak = np.argmax(corr[start:]) + start
    i_interp = parabolic(corr, i_peak)[0]

    return fs / i_interp


def freq_from_hps(signal, fs):
    """Estimate frequency using harmonic product spectrum
    Low frequency noise piles up and overwhelms the desired peaks
    """
    N = len(signal)
    signal -= np.mean(signal)  # Remove DC offset

    # Compute Fourier transform of windowed signal
    windowed = signal * kaiser(N, 100)

    # Get spectrum
    X = np.log(abs(np.fft.rfft(windowed)))

    # Downsample sum logs of spectra instead of multiplying
    hps = np.copy(X)
    for h in range(2, 9):  # TODO: choose a smarter upper limit
        dec = decimate(X, h)
        hps[: len(dec)] += dec

    # Find the peak and interpolate to get a more accurate peak
    i_peak = np.argmax(hps[: len(dec)])
    i_interp = parabolic(hps, i_peak)[0]

    # Convert to equivalent frequency
    return fs * i_interp / N  # Hz


# {Dynamic time warping}#


def simple_complexity(data):
    """Calculate the complexity of a series of data by calculating the
    length of the square root of the sume of the squares of the
    v-yalues for the signal.
    Inputs
    ------

        data: A Numpy array-like object
    Output
    ------
        complexity: A float value representing the complexity of the data.
    Examples
    --------
    """
    return np.sqrt(np.sum(np.diff(data) ** 2))


def complexity_correction_factor(t1, t2):
    complexities = [simple_complexity(t) for t in [t1, t2]]
    return max(complexities) / min(complexities)


def euclidean(t1, t2, **kwargs):
    """
    * ---------------{Function}---------------
    * Computes the Euclidean distance between two vectors using the numpy library.
    * ----------------{Returns}---------------
    * -> distance    ::float | The Euclidean distance between the two input vectors.
    * ----------------{Params}----------------
    * : t1           ::array | The first input vector.
    * : t2           ::array | The second input vector.
    * : **kwargs     ::dict  | Additional keyword arguments to be passed to the function (not used in this implementation).
    * ----------------{Usage}-----------------
    * >>> euclidean([1, 2, 3], [4, 5, 6])
    * 5.196152422706632
    """
    return np.sqrt(np.sum(np.abs(t1**2 - t2**2)))


# ------{From High Energy Physics}------#

# {http://arxiv.org/pdf/1101.0390v2.pdf}#

# ----{BumpHunter}---#

from math import log

from numpy import array, random
from scipy.stats import percentileofscore, poisson


def evaluate_statistic(data, mc, verbose=False, edges=None):
    """
        * type-def ::[Array] ::[Array] ::Bool ::[Array] -> Tuple[float, Tuple[int, int]]
    * ---------------{Function}---------------
        * Evaluates the bumphunter statistic for given data and Monte Carlo values, returning the minimum p-value and its corresponding interval.
    * ----------------{Returns}---------------
        * -> min_pvalue_log ::float | The negative logarithm of the minimum p-value found
        * -> (lo, hi)       ::Tuple[int, int] | The lower and upper bin indices of the interval corresponding to the minimum p-value
    * ----------------{Params}----------------
        * : data       ::Array[float] | The array of input data values
        * : mc         ::Array[float] | The array of Monte Carlo values
        * : verbose    ::bool | Optional; if True, print additional information (default: False)
        * : edges      ::Array[float] | Optional; the bin edges of the input histograms (default: None)
    * ----------------{Usage}-----------------
        * >>> evaluate_statistic(data, mc)
        * (-log(min_pvalue), (lo, hi))
    * ----------------{Notes}-----------------
        * This function computes the p-value for all possible windows within the search range.
        * The search range is determined by the first and last non-zero bins of the Monte Carlo values.
        * The function uses the `poisson.cdf()` function from the `scipy.stats` library to compute p-values.
    """
    # Get search range (first bin with data, last bin with data)
    (nzi,) = mc.nonzero()  # nzi = non-zero indices
    search_lo, search_hi = nzi[0], nzi[-1]

    def all_windows():
        """
            * type-def ::None -> Iterator[Tuple[int, int]]
        * ---------------{Function}---------------
            * Iterator that yields tuples of (lo, hi) bin indices for all possible windows within the search range.
        * ----------------{Yields}---------------
            * -> (lo, hi) ::Tuple[int, int] | The lower and upper bin indices of a window
        * ----------------{Usage}-----------------
            * >>> for lo, hi in all_windows():
            * ...     # Process the window from lo to hi
        * ----------------{Notes}-----------------
            * The window sizes range from one bin up to half of the full search range.
            * The step size is half the binwidth.
        """
        # Try windows from one bin in width up to half of the full range
        min_win_size, max_win_size = 1, (search_hi - search_lo) // 2
        for binwidth in range(min_win_size, max_win_size):
            if verbose:
                print(" --- binwidth = ", binwidth)
            step = max(1, binwidth // 2)  # Step size <- half binwidth
            for pos in range(search_lo, search_hi - binwidth, step):
                yield pos, pos + binwidth

    def pvalue(lo, hi):
        """
            * type-def ::Int ::Int -> float
        * ---------------{Function}---------------
            * Computes the p-value for the given window [lo, hi) using the data and Monte Carlo values.
        * ----------------{Returns}---------------
            * -> p ::float | The computed p-value for the given window
        * ----------------{Params}----------------
            * : lo ::int | The lower bin index of the window
            * : hi ::int | The upper bin index of the window
        * ----------------{Usage}-----------------
            * >>> p = pvalue(3, 5)
            * >>> print(p)
            * 0.12345
        * ----------------{Notes}-----------------
            * If the Monte Carlo prediction for the window is zero, the function asserts that the data is also zero.
            * If the data value is less than the Monte Carlo value, the function returns a p-value of 1 (ignoring dips).
            * The function uses the `poisson.cdf()` function from the `scipy.stats` library to compute p-values.
        """
        d, m = data[lo:hi].sum(), mc[lo:hi].sum()
        if m == 0:
            # MC prediction is zero. Not sure what then..
            assert d == 0, "Data = {0} where the prediction is zero..".format(d)
            return 1
        if d < m:
            return 1  # "Dips" get ignored.

        # P(d >= m)
        p = 1 - poisson.cdf(d - 1, m)

        if verbose and edges:
            print(
                "{0:2} {1:2} [{2:8.3f}, {3:8.3f}] {4:7.0f} {5:7.3f} {6:.5f} {7:.2f}".format(
                    lo, hi, edges[lo], edges[hi], d, m, p, -log(p)
                )
            )

        return p

    min_pvalue, (lo, hi) = min((pvalue(lo, hi), (lo, hi)) for lo, hi in all_windows())

    return -log(min_pvalue), (lo, hi)


def make_toys(prediction, n):
    """
        * type-def ::[Array] ::Int -> [Array]'
    * ---------------{Function}---------------
        * Fluctuates the input `prediction` distribution `n` times using Poisson distribution.
    * ----------------{Returns}---------------
        * -> fluctuated_predictions ::Array[float] | The array of fluctuated predictions
    * ----------------{Params}----------------
        * : prediction  ::Array[float] | The input distribution to fluctuate
        * : n           ::int | The number of times to fluctuate the input distribution
    * ----------------{Usage}-----------------
        * >>> make_toys(prediction, 100)
        * array([[fluctuated_prediction_1], [fluctuated_prediction_2], ...])
    * ----------------{Notes}-----------------
        * This function uses the `random.mtrand.poisson()` function from the `numpy` library to generate Poisson-distributed random fluctuations of the input distribution.
    """
    return random.mtrand.poisson(prediction, size=(n, len(prediction)))


def bumphunter(hdata, hmc, n):
    """
        * type-def ::[Array] ::[Array] ::Int -> Tuple[float, Tuple[float, float], List[float], float, float]
    * ---------------{Function}---------------
        * Computes the bumphunter statistic, runs `n` pseudo-experiments, and returns the measurement, interval, and p-value.
    * ----------------{Returns}---------------
        * -> measurement        ::float | The bumphunter statistic measurement
        * -> (lo, hi)           ::Tuple[float, float] | The lower and upper bounds of the interval
        * -> pseudo_experiments ::List[float] | The list of bumphunter statistic values from the pseudo-experiments
        * -> pvalue             ::float | The computed p-value
        * -> pvalue_uncertainty ::float | The uncertainty of the p-value
    * ----------------{Params}----------------
        * : hdata       ::Array[float] | The array of input data values
        * : hmc         ::Array[float] | The array of Monte Carlo values
        * : n           ::int | The number of pseudo-experiments to run
    * ----------------{Usage}-----------------
        * >>> bumphunter(hdata, hmc, 1000)
        * (measurement, (lo, hi), pseudo_experiments, pvalue, pvalue_uncertainty)
    * ----------------{Notes}-----------------
        * This function requires the following functions to be defined elsewhere:
            * `evaluate_statistic()` to compute the bumphunter statistic and interval for given data and Monte Carlo values
            * `make_toys()` to generate `n` pseudo-experiments from the Monte Carlo values
        * `percentileofscore()` from the `scipy.stats` library is used to compute the p-value from the pseudo-experiments and measurement.
    """
    data = array([hdata[i] for i in range(1, hdata.GetNbinsX())])
    mc = array([hmc[i] for i in range(1, hmc.GetNbinsX())])

    pseudo_experiments = [evaluate_statistic(pe, mc)[0] for pe in make_toys(mc, n)]

    measurement, (lo, hi) = evaluate_statistic(data, mc)

    pvalue = 1.0 - (percentileofscore(pseudo_experiments, measurement) / 100.0)
    pvalue_uncertainty = sqrt(pvalue * (1.0 - pvalue) / n)

    return measurement, (lo, hi), pseudo_experiments, pvalue, pvalue_uncertainty


def remove_outliers(data):
    """
        * type-def ::[List] ::[x'∈X] ::[x'∈X] -> [List]'
    * ---------------{Function}---------------
        * Removes outliers from a list of data points by replacing them with 0.
    * ----------------{Returns}---------------
        * -> result    ::List[float] | The list with outliers replaced by 0
    * ----------------{Params}----------------
        * : data       ::List[float] | The list of data points to process
    * ----------------{Usage}-----------------
        * >>> remove_outliers([1, 2, 3, 100, 4, 5])
        * [1, 2, 3, 0, 4, 5]
    * ----------------{Notes}-----------------
        * Outliers are determined based on the mean and standard deviation of the data.
        * Data points more than one standard deviation away from the mean are considered outliers.
        * This function requires the `stddev()` function to be defined elsewhere, which computes the standard deviation of a list of values.
    """
    mean = sum(data) / len(data)
    std = stddev(data)

    result = []
    for i in data:
        if math.fabs((i - mean)) > 1 * std:
            result.append(0)
        else:
            result.append(i)
    return result


def rescale(values, new_min=0, new_max=100):
    """
        * type-def ::[List] ::[x'∈X] ::Num ::Num -> [List]'
    * ---------------{Function}---------------
        * Rescales a list of values to a new range.
    * ----------------{Returns}---------------
        * -> output    ::List[float] | The list of rescaled values
    * ----------------{Params}----------------
        * : values     ::List[float] | The list of values to rescale
        * : new_min    ::float | The new minimum value for the rescaled list (default: 0)
        * : new_max    ::float | The new maximum value for the rescaled list (default: 100)
    * ----------------{Usage}-----------------
        * >>> rescale([10, 20, 30], 0, 1)
        * [0.0, 0.5, 1.0]
    * ----------------{Notes}-----------------
        * This function linearly rescales the input values to the specified range.
        * The new range is determined by the `new_min` and `new_max` parameters.
    """
    output = []
    old_min, old_max = min(values), max(values)

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return output
