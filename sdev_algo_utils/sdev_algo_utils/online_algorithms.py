""" Online Algorithms implementations for python"""

import numpy as np
from random import shuffle
import pandas as pd

# from detections.aimbotlstm.utils.enums import Logging
# from detections.aimbotlstm.utils.pandas_utils import get_subset
import math
import json
import types
import textwrap
import collections

iterator_types = (types.GeneratorType, collections.Iterable)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value


##==========================={Running mean|variance|stdev}======================


class RunningStats:
    """
    O(1) space complexity Running Statistics (mean, variance, stdev) based on
    1962 paper by B. P. Welford and is presented
    in Donald Knuth’s Art of Computer Programming, Vol 2, page 232,
    """

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0
        self.min_ = float("+inf")
        self.max_ = float("-inf")

    def clear(self):
        self.n = 0

    def push(self, x, *args):
        self.n += 1
        self.min_ = min(self.min_, x)
        self.max_ = max(self.max_, x)

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return math.sqrt(self.variance())

    def max_num(self):
        return self.max_

    def min_num(self):
        return self.min_


##==========================={Running mean|variance|stdev|kurtosis|skewness}====


class RunningStatsSK:
    """
    Modified version to compute kurtosis and skewness in one pass
    O(1) space complexity Running Statistics (mean, variance, stdev) based on
    1962 paper by B. P. Welford and is presented
    in Donald Knuth’s Art of Computer Programming, Vol 2, page 232,
    """

    def __init__(self):
        self.n = 0
        self.M1 = 0
        self.M2 = 0
        self.M3 = 0
        self.M4 = 0

    def clear(self):
        self.n = 0

    def push(self, x, *args):
        self.n1 = self.n
        self.n += 1
        self.delta = x - self.M1
        self.delta_n = self.delta / self.n
        self.delta_n2 = self.delta_n * self.delta_n
        self.term1 = self.delta * self.delta_n * self.n1
        self.M1 += self.delta_n
        self.M4 += (
            self.term1 * self.delta_n2 * (self.n * self.n - 3 * self.n + 3)
            + 6 * self.delta_n2 * self.M2
            - 4 * self.delta_n * self.M3
        )
        self.M3 += self.term1 * self.delta_n * (self.n - 2) - 3 * self.delta_n * self.M2
        self.M2 = self.term1

    def NumDataValues(self):
        return self.n

    def mean(self):
        return self.M1

    def variance(self):
        return self.M2 / (self.n - 1.0)

    def standard_deviation(self):
        return np.sqrt(self.variance())

    def skewness(self):
        return np.sqrt(self.n) * self.M3 / np.power(self.M2, 1.5)

    def kurtosis(self):
        return self.n * self.M4 / (self.M2 * self.M2) - 3.0


##==========================={Median from bins O(1) space}======================


class BinMedian:
    """
    adapted from http://www.stat.cmu.edu/~ryantibs/median/binmedian.f
    """

    def __init__(self, data):
        n = len(data)
        x = data
        self.sigma = self.binmedian(n, x)

    def binmedian(self, n, x):
        sum = 0
        for i in range(n):
            sum += x[i]
        mu = sum / n
        sum = 0
        for i in range(n):
            sum += (x[i] - mu) * (x[i] - mu)
        sigma = np.sqrt(sum / n)
        ## Bin x across the interval [mu-sigma, mu+sigma]
        bottomcount = 0
        bincounts = list(np.zeros(1001, int))
        scalefactor = 1000 / (2 * sigma)
        leftend = mu - sigma
        rightend = mu + sigma
        for i in range(n):
            if x[i] < leftend:
                bottomcount += 1
            elif x[i] < rightend:
                bin = int((x[i] - leftend) * scalefactor)
                bincounts[bin] += 1
        # if n is odd
        if n & 1:
            # Recursive step ############################
            # int k, r, count, medbin;                  #
            # float oldscalefactor, oldleftend          #
            # int oldbin                                #
            # float temp                                #
            #############################################
            k = (n + 1) // 2
            r = 0
            while True:
                # Find the bin that contains the median, and the order
                # of the median within that bin
                count = bottomcount
                for i in range(1001):
                    count += bincounts[i]
                    if count >= k:
                        medbin = i
                        k = k - (count - bincounts[i])
                        break
                bottomcount = 0
                for i in range(1001):
                    bincounts[i] = 0
                oldscalefactor = scalefactor
                oldleftend = leftend
                scalefactor = 1000 * oldscalefactor
                leftend = medbin / oldscalefactor + oldleftend
                rightend = (medbin + 1) / oldscalefactor + oldleftend

                # Determine which points map to medbin and put them in
                # spots r,...n-1
                i = r
                r = n
                while i < r:
                    oldbin = int((x[i] - oldleftend) * oldscalefactor)
                    if oldbin == medbin:
                        r -= 1
                        x[i], x[r] = x[r], x[i]

                        # Re-Bin on a finer scale
                        if x[i] < leftend:
                            bottomcount += 1
                        elif x[i] < rightend:
                            bin = int((x[i] - leftend) * scalefactor)
                            bincounts[bin] += 1
                    else:
                        i += 1

                # Stop if all points in medbin are the same
                samepoints = 1
                i = r + 1
                while i < n:

                    if x[i] != x[r]:
                        samepoints = 0
                        break
                    if samepoints:
                        return x[r]
                    # stop if there's <= 20 points left
                    if n - r <= 20:
                        break
                    i += 1
                # Perform insertion sort on the remaining points
                # and then pick the kth smallest
                i = r + 1
                while i < n:
                    a = x[i]
                    j = i - 1
                    while j >= r:
                        j -= 1
                        if x[j] > a:
                            break
                        x[j + 1] = x[j]
                    x[j + 1] = a
                    i += 1

                # return k, r , x
                return x[r - 1 + k]

        else:
            x.append(0)
            # return 0
            return self.binmedian(n + 1, x)


##==========================={t-digest quantiles}===============================


class TDigest(object):
    def __init__(self, delta=0.01, compression=20):
        self.delta = float(delta)
        self.compression = compression
        self.tdc = TDigestCore(self.delta)

    def push(self, x, w):
        self.tdc.push(x, w)
        if len(self) > self.compression / self.delta:
            self.compress()

    def compress(self):
        aux_tdc = TDigestCore(self.delta)
        centroid_list = self.tdc.centroid_list
        shuffle(centroid_list)
        for c in centroid_list:
            aux_tdc.push(c.mean, c.count)
        self.tdc = aux_tdc

    def quantile(self, x):
        return self.tdc.quantile(x)

    def serialize(self):
        centroids = [[c.mean, c.count] for c in self.tdc.centroid_list]
        return json.dumps(centroids)

    def __len__(self):
        return len(self.tdc)

    def __repr__(self):
        return str(self.tdc)


class Centroid(object):
    def __init__(self, x, w, id):
        self.mean = float(x)
        self.count = float(w)
        self.id = id

    def push(self, x, w):
        self.count += w
        self.mean += w * (x - self.mean) / self.count

    def equals(self, c):
        if c.id == self.id:
            return True
        else:
            return False

    def distance(self, x):
        return abs(self.mean - x)

    def __repr__(self):
        return "Centroid{mean=%.1f, count=%d}" % (self.mean, self.count)


class TDigestCore(object):
    def __init__(self, delta):
        self.delta = delta
        self.centroid_list = []
        self.n = 0
        self.id_counter = 0

    def push(self, x, w):
        self.n += 1

        if self.centroid_list:
            S = self._closest_centroids(x)
            shuffle(S)
            for c in S:
                if w == 0:
                    break
                q = self._centroid_quantile(c)
                delta_w = min(4 * self.n * self.delta * q * (1 - q) - c.count, w)
                c.push(x, delta_w)
                w -= delta_w

        if w > 0:
            self.centroid_list.append(Centroid(x, w, self.id_counter))
            self.centroid_list.sort(key=lambda c: c.mean)
            self.id_counter += 1

    def quantile(self, x):
        if len(self.centroid_list) < 3:
            return 0.0
        total_weight = sum([centroid.count for centroid in self.centroid_list])
        q = x * total_weight
        m = len(self.centroid_list)
        cumulated_weight = 0
        for nr in range(m):
            current_weight = self.centroid_list[nr].count
            if cumulated_weight + current_weight > q:
                if nr == 0:
                    delta = (
                        self.centroid_list[nr + 1].mean - self.centroid_list[nr].mean
                    )
                elif nr == m - 1:
                    delta = (
                        self.centroid_list[nr].mean - self.centroid_list[nr - 1].mean
                    )
                else:
                    delta = (
                        self.centroid_list[nr + 1].mean
                        - self.centroid_list[nr - 1].mean
                    ) / 2
                return (
                    self.centroid_list[nr].mean
                    + ((q - cumulated_weight) / (current_weight) - 0.5) * delta
                )
            cumulated_weight += current_weight
        return self.centroid_list[nr].mean

    def _closest_centroids(self, x):
        S = []
        z = None
        for centroid in self.centroid_list:
            d = centroid.distance(x)
            if z is None:
                z = d
                S.append(centroid)
            elif z == d:
                S.append(centroid)
            elif z > d:
                S = [centroid]
                z = d
            elif x > centroid.mean:
                break
        T = []
        for centroid in S:
            q = self._centroid_quantile(centroid)
            if centroid.count + 1 <= 4 * self.n * self.delta * q * (1 - q):
                T.append(centroid)
        return T

    def _centroid_quantile(self, c):
        q = 0
        for centroid in self.centroid_list:
            if centroid.equals(c):
                q += c.count / 2
                break
            else:
                q += centroid.count
        return q / sum([centroid.count for centroid in self.centroid_list])

    def __len__(self):
        return len(self.centroid_list)

    def __repr__(self):
        return "[ %s ]" % ", ".join([str(c) for c in self.centroid_list])


##==========================={Stream Histogram}=================================


##-------------{Stream functions}-------
from math import sqrt


def argmin(array):
    # http://lemire.me/blog/archives/2008/12/17/fast-argmax-in-python/
    return array.index(min(array))


def bin_diff(array, weighted=False):
    return [_diff(a, b, weighted) for a, b in zip(array[:-1], array[1:])]


def _diff(a, b, weighted):
    diff = b.value - a.value
    if weighted:
        diff *= log(_E + min(a.count, b.count))
    return diff


def bin_sums(array, less=None):
    return [
        (a.count + b.count) / 2.
        for a, b in zip(array[:-1], array[1:])
        if less is None or b.value <= less
    ]


def accumulate(iterable):
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total += element
        yield total


def roots(a, b, c):
    """Super simple quadratic solver."""
    d = b ** 2.0 - (4.0 * a * c)
    if d < 0:
        raise (ValueError("This equation has no real solution!"))
    elif d == 0:
        x = (-b + sqrt(d)) / (2.0 * a)
        return (x, x)
    else:
        x1 = (-b + sqrt(d)) / (2.0 * a)
        x2 = (-b - sqrt(d)) / (2.0 * a)
        return (x1, x2)


##-------------{Core}-------------------

import sys
from bisect import bisect_left

from sortedcontainers import SortedListWithKey

_all__ = ["StreamHist", "Bin"]


class StreamHist(object):
    """A StreamHist implementation."""

    def __init__(self, maxbins=64, weighted=False, freeze=None):
        """Create a Histogram with a max of n bins."""
        super(StreamHist, self).__init__()
        # self.bins = []
        self.bins = SortedListWithKey(key=lambda b: b.value)
        self.maxbins = maxbins  # A useful property
        self.total = 0
        self.weighted = weighted
        self._min = None  # A useful property
        self._max = None  # A useful property
        self.freeze = freeze
        self.missing_count = 0

    def update(self, n, count=1):
        """Add a point to the histogram."""
        if n is None:
            # We simply keep a count of the number of missing values
            self.missing_count += count
            return self
        if isinstance(n, iterator_types):
            # Shortcut for updating a histogram with an iterable
            # This works for anything that supports iteration, including
            # file-like objects and readers
            # This also means that nested lists (and similar structures) will
            # be 'unpacked' and added to the histogram 'automatically'
            for p in n:
                self.update(p, count)  # Count is assumed to apply for all
        else:
            self.insert(n, count)
        return self.trim()

    def insert(self, n, count):
        """Inserts a point to the histogram.
        This method implements Steps 1-4 from Algorithm 1 (Update) in ref [1].
        Notes
        -----
        It is better to use `update` when inserting data into the histogram,
        as `insert` does not automatically update the total point count, or
        call `trim` after the insertion. For large batches of inserts, insert
        may be more efficient, but you are responsible for updating counts
        and trimming the bins 'manually'.
        Examples
        --------
        >>> # Using insert
        >>> h = StreamHist().insert(1).insert(2).insert(3)
        >>> h.update_total(3)
        >>> h.trim()
        >>> # Using update
        >>> h = StreamHist().update([1, 2, 3])
        """
        self.update_total(count)
        if self._min is None or self._min > n:
            self._min = n
        if self._max is None or self._max < n:
            self._max = n
        b = Bin(value=n, count=count)
        if b in self.bins:
            index = self.bins.index(b)
            self.bins[index].count += count
        else:
            if self.freeze is not None and self.total >= self.freeze:
                index = self.bins.bisect(Bin(n, count))
                if index:
                    prev_dist = n - self.bins[index - 1].value
                else:
                    prev_dist = sys.float_info.max
                if index and index < len(self.bins):
                    next_dist = self.bins[index].value - n
                else:
                    next_dist = sys.float_info.max
                if prev_dist < next_dist:
                    self.bins[index - 1].count += count
                else:
                    self.bins[index].count += count
            else:
                self.bins.add(b)

    def cdf(self, x):
        """Return the value of the cumulative distribution function at x."""
        return self.sum(x) / self.total

    def pdf(self, x):
        """Return the value of the probability density function at x."""
        return self.density(x) / self.total

    def bounds(self):
        """Return the upper (max( and lower (min) bounds of the distribution."""
        if len(self):
            return (self._min, self._max)
        return (None, None)

    def count(self):
        """Return the number of bins in this histogram."""
        return self.total

    def median(self):
        """Return a median for the points inserted into the histogram.
        This will be the true median whenever the histogram has less than
        the maximum number of bins, otherwise it will be an approximation.
        """
        try:
            if self.total == 0:
                return None
            if len(self.bins) >= self.maxbins:
                # Return the approximate median
                return self.quantiles(0.5)[0]
            else:
                # Return the 'exact' median when possible
                mid = (self.total) / 2
                if self.total % 2 == 0:
                    return (self.bins[mid - 1] + self.bins[mid]).value
                return self.bins[mid].value
        except:
            return None

    def mean(self):
        """Return the sample mean of the distribution."""
        if self.total == 0:
            return None
        s = 0.0  # Sum
        for b in self.bins:
            s += b.value * b.count
        return s / float(self.total)

    def var(self):
        """Return the variance of the distribution."""
        if self.total < 2:
            return None
        s = 0.0
        m = self.mean()  # Mean
        for b in self.bins:
            s += b.count * (b.value - m) ** 2
        return s / float(self.total)

    def min(self):
        """Return the minimum value in the histogram."""
        return self._min

    def max(self):
        """Return the maximum value in the histogram."""
        return self._max

    def trim(self):
        """Merge adjacent bins to decrease bin count to the maximum value.
        This method implements Steps 5-6 from Algorithm 1 (Update) in ref [1].
        """
        while len(self.bins) > self.maxbins:
            index = argmin(bin_diff(self.bins, self.weighted))
            prv = self.bins.pop(index)
            self.bins[index] += prv
        return self

    def scale_down(self, exclude):
        pass  # By default, we do nothing

    def __str__(self):
        """Return a string reprentation of the histogram."""
        if len(self.bins):
            string = "Mean\tCount\n----\t-----\n"
            for b in self.bins:
                string += "%d\t%i\n" % (b.value, b.count)
            string += "----\t-----\n"
            string += "Missing values: %s\n" % self.missing_count
            string += "Total count: %s" % self.total
            return string
        return "Empty histogram"

    def to_dict(self):
        """Return a dictionary representation of the histogram."""
        bins = list()
        for b in self.bins:
            bins.append({"mean": b.value, "count": b.count})
        info = dict(
            missing_count=self.missing_count,
            maxbins=self.maxbins,
            weighted=self.weighted,
            freeze=self.freeze,
        )
        return dict(bins=bins, info=info)

    @classmethod
    def from_dict(cls, d):
        """Create a StreaHist object from a dictionary representation.
        The dictionary must be in the format given my `to_dict`. This class
        method, combined with the `to_dict` instance method, can facilitate
        communicating StreamHist objects across processes or networks.
        """
        info = d["info"]
        bins = d["bins"]
        hist = cls(info["maxbins"], info["weighted"], info["freeze"])
        hist.missing_count = info["missing_count"]
        for b in bins:
            count = b["count"]
            value = b["mean"]
            hist.bins.append(Bin(value, count))
        return hist

    def __len__(self):
        """Return the number of bins in this histogram."""
        return len(self.bins)

    def update_total(self, size=1):
        """Update the internally-stored total number of points."""
        self.total += size

    def __add__(self, other):
        """Merge two StreamHist objects into one."""
        res = self.copy()
        return res.merge(other)

    def __iadd__(self, other):
        """Merge another StreamHist object into this one."""
        return self.merge(other)

    def __radd__(self, other):
        """Reverse merge two objects.
        This is useful for merging a list of histograms via sum or similar.
        """
        return self + other

    def merge(self, other, size=None):
        """Merge another StreamHist object into this one.
        This method implements Algorithm 2 (Merge) in ref [1].
        """
        if other == 0:  # Probably using sum here...
            return self  # This is a little hacky...
        for b in other.bins:
            self.bins.add(b)
        self.total += other.total
        if size is not None:
            self.maxbins = size
        self.trim()
        if self._min is None:
            self._min = other._min
        else:
            if other._min is not None:
                self._min = min(self._min, other._min)
        if self._max is None:
            self._max = other._max
        else:
            if other._max is not None:
                self._max = max(self._max, other._max)
        self.missing_count += other.missing_count
        return self

    def copy(self):
        """Make a deep copy of this histogram."""
        res = type(self)(int(self.maxbins), bool(self.weighted))
        res.bins = self.bins.copy()
        res._min = float(self._min) if self._min is not None else None
        res._max = float(self._max) if self._max is not None else None
        res.total = int(self.total)
        res.missing_count = int(self.missing_count)
        res.freeze = int(self.freeze) if self.freeze is not None else None
        return res

    def describe(self, quantiles=[0.25, 0.50, 0.75]):
        """Generate various summary statistics."""
        data = [self.count(), self.mean(), self.var(), self.min()]
        data += self.quantiles(*quantiles) + [self.max()]
        names = ["count", "mean", "var", "min"]
        names += ["%i%%" % round(q * 100., 0) for q in quantiles] + ["max"]
        return dict(zip(names, data))

    def compute_breaks(self, n=50):
        """Return output like that of numpy.histogram."""
        last = 0.0
        counts = []
        bounds = linspace(*self.bounds(), num=n)
        for e in bounds[1:]:
            new = self.sum(e)
            counts.append(new - last)
            last = new
        return counts, bounds

    def print_breaks(self, num=50):
        """Print a string reprentation of the histogram."""
        string = ""
        for c, b in zip(*self.compute_breaks(num)):
            bar = str()
            for i in range(int(c / float(self.total) * 200)):
                bar += "."
            string += str(b) + "\t" + bar + "\n"
        print(string)

    def sum(self, x):
        """Return the estimated number of points in the interval [  , b]."""
        x = float(x)
        if x < self._min:
            ss = 0.0  # Sum is zero!
        elif x >= self._max:
            ss = float(self.total)
        elif x == self.bins[-1].value:
            # Shortcut for when i == max bin (see Steps 3-6)
            last = self.bins[-1]
            ss = float(self.total) - (float(last.count) / 2.0)
        # elif x <= self.bins[0].value:
        #     # Shortcut for when i == min bin (see Steps 3-6)
        #     first = self.bins[0]
        #     ss = float(first.count) / 2.0
        else:
            bin_i = self.floor(x)
            if bin_i is None:
                bin_i = Bin(value=self._min, count=0)
            bin_i1 = self.higher(x)
            if bin_i1 is None:
                bin_i1 = Bin(value=self._max, count=0)
            if bin_i.value == self._min:
                prev_sum = self.bins[0].count / 2.0
            else:
                temp = bin_sums(self.bins, less=x)
                if len(temp):
                    prev_sum = sum(temp)
                else:
                    prev_sum = 0.0
            ss = _compute_sum(x, bin_i, bin_i1, prev_sum)
        return ss

    def density(self, p):
        p = float(p)
        if p < self._min or p > self._max:
            dd = 0.0
        elif p == self._min and p == self._max:
            dd = float("inf")
        elif Bin(value=p, count=0) in self.bins:
            high = next_after(p, float("inf"))
            low = next_after(p, -float("inf"))
            dd = (self.density(low) + self.density(high)) / 2.0
        else:
            bin_i = self.lower(p)
            if bin_i is None:
                bin_i = Bin(value=self._min, count=0)
            bin_i1 = self.higher(p)
            if bin_i1 is None:
                bin_i1 = Bin(value=self._max, count=0)
            dd = _compute_density(p, bin_i, bin_i1)
        return dd

    def quantiles(self, *quantiles):
        """Return the estimated data value for the given quantile(s).
        The requested quantile(s) must be between 0 and 1. Note that even if a
        single quantile is input, a list is always returned.
        """
        temp = bin_sums(self.bins)
        sums = list(accumulate(temp))
        result = []
        for x in quantiles:
            target_sum = x * self.total
            if x <= 0:
                qq = self._min
            elif x >= self.total:
                qq = self._max
            else:
                index = bisect_left(sums, target_sum)
                bin_i = self.bins[index]
                if index < len(sums):
                    bin_i1 = self.bins[index + 1]
                else:
                    bin_i1 = self.bins[index]
                if index:
                    prev_sum = sums[index - 1]
                else:
                    prev_sum = 0.0
                qq = _compute_quantile(target_sum, bin_i, bin_i1, prev_sum + 1)
            result.append(qq)
        return result

    def floor(self, p):
        hbin = Bin(p, 0)
        index = self.bins.bisect_left(hbin)
        if hbin not in self.bins:
            index -= 1
        return self.bins[index] if index >= 0 else None

    def ceiling(self, p):
        hbin = Bin(p, 0)
        index = self.bins.bisect_right(hbin)
        if hbin in self.bins:
            index -= 1
        return self.bins[index] if index < len(self.bins) else None

    def lower(self, p):
        index = self.bins.bisect_left(Bin(p, 0)) - 1
        return self.bins[index] if index >= 0 else None

    def higher(self, p):
        index = self.bins.bisect_right(Bin(p, 0))
        return self.bins[index] if index < len(self.bins) else None


# Utility functions (should not be included in __all__)


def _compute_density(p, bin_i, bin_i1):
    """Finding the density starting from the sum.
    s = p + (1/2 + r - r^2/2)*i + r^2/2*i1
    r = (x - m) / (m1 - m)
    s_dx = i - (i1 - i) * (x - m) / (m1 - m)
    """
    b_diff = p - bin_i.value
    p_diff = bin_i1.value - bin_i.value
    bp_ratio = b_diff / p_diff

    inner = (bin_i1.count - bin_i.count) * bp_ratio
    return (bin_i.count + inner) * (1.0 / (bin_i1.value - bin_i.value))


def _compute_quantile(x, bin_i, bin_i1, prev_sum):
    try:
        d = x - prev_sum
        a = bin_i1.count - bin_i.count
        if a == 0:
            offset = d / ((bin_i.count + bin_i1.count) / 2.0)
            u = bin_i.value + (offset * (bin_i1.value - bin_i.value))
        else:
            b = 2.0 * bin_i.count
            c = -2.0 * d
            z = _find_z(a, b, c)
            u = bin_i.value + (bin_i1.value - bin_i.value) * z
        return u
    except:
        return None


def _compute_sum(x, bin_i, bin_i1, prev_sum):
    b_diff = x - bin_i.value
    p_diff = bin_i1.value - bin_i.value
    bp_ratio = b_diff / p_diff

    i1Term = 0.5 * bp_ratio ** 2.0
    iTerm = bp_ratio - i1Term

    first = prev_sum + bin_i.count * iTerm
    ss = first + bin_i1.count * i1Term
    return ss


def _find_z(a, b, c):
    result_root = None
    candidate_roots = roots(a, b, c)
    for candidate_root in candidate_roots:
        if candidate_root >= 0 and candidate_root <= 1:
            result_root = candidate_root
            break
    return result_root


class Bin(object):
    """Histogram bin object.
    This class implements a simple (value, count) histogram bin pair with
    several added features such as the ability to merge two bins, comparison
    methods, and the ability to export and import from dictionaries . The Bin
    class should be used in conjunction with the StreamHist.
    """

    __slots__ = ["value", "count"]

    def __init__(self, value, count=1):
        """Create a Bin with a given mean and count.
        Parameters
        ----------
        value : float
            The mean of the bin.
        count : int (default=1)
            The number of points in this bin. It is assumed that there are
            `count` points surrounding `value`, of which `count/2` points are
            to the left and `count/2` points are to the right.
        """
        super(Bin, self).__init__()
        self.value = value
        self.count = count

    @classmethod
    def from_dict(cls, d):
        """Create a bin instance from a dictionary.
        Parameters
        ----------
        d : dict
            The dictionary must at a minimum a `mean` or `value` key. In
            addition, it may contain a `count` key which contains the number
            of points in the bin.
        """
        value = d.get("mean", d.get("value", None), None)
        if value is None:
            raise ValueError("Dictionary must contain a mean or value key.")
        return cls(value=value, count=d.get("count", 1))

    def __getitem__(self, index):
        """Alternative method for getting the bin's mean and count.
        Parameters
        ----------
        index : int
            The index must be either 0 or 1, where 0 gets the mean (value),
            and 1 gets the count.
        """
        if index == 0:
            return self.value
        elif index == 1:
            return self.count
        raise IndexError("Invalid index (must be 0 or 1).")

    def __repr__(self):
        """Simple representation of a histogram bin.
        Returns
        -------
        Bin(value=`value`, count=`count`) where value and count are the bin's
        stored mean and count.
        """
        return "Bin(value=%d, count=%d)" % (self.value, self.count)

    def __iter__(self):
        """Iterator over the mean and count of this bin."""
        yield ("mean", self.value)
        yield ("count", self.count)

    def __str__(self):
        """String representation of a histogram bin."""
        return str(dict(self))

    def __eq__(self, obj):
        """Tests for equality of two bins.
        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value == obj.value

    def __lt__(self, obj):
        """Tests if this bin has a lower mean than another bin.
        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value < obj.value

    def __gt__(self, obj):
        """Tests if this bin has a higher mean than another bin.
        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value > obj.value

    def __add__(self, obj):
        """Merge this bin with another bin and return the result.
        This method implements Step 7 from Algorithm 1 (Update) in ref [1].
        Parameters
        ----------
        obj : Bin
            The bin that will be merged with this bin.
        """
        count = float(self.count + obj.count)  # Summed heights
        if count:
            # Weighted average
            value = self.value * float(self.count) + obj.value * float(obj.count)
            value /= count
        else:
            value = 0.0
        return Bin(value=value, count=int(count))

    def __iadd__(self, obj):
        """Merge another bin into this one.
        Parameters
        ----------
        obj : Bin
            The bin that will be merged into this bin.
        """
        out = self + obj
        self = out
        return self


##-------------{Bin Histogram}----------


class Bin(object):
    """Histogram bin object.
    This class implements a simple (value, count) histogram bin pair with
    several added features such as the ability to merge two bins, comparison
    methods, and the ability to export and import from dictionaries . The Bin
    class should be used in conjunction with the StreamHist.
    """

    __slots__ = ["value", "count"]

    def __init__(self, value, count=1):
        """Create a Bin with a given mean and count.
        Parameters
        ----------
        value : float
            The mean of the bin.
        count : int (default=1)
            The number of points in this bin. It is assumed that there are
            `count` points surrounding `value`, of which `count/2` points are
            to the left and `count/2` points are to the right.
        """
        super(Bin, self).__init__()
        self.value = value
        self.count = count

    @classmethod
    def from_dict(cls, d):
        """Create a bin instance from a dictionary.
        Parameters
        ----------
        d : dict
            The dictionary must at a minimum a `mean` or `value` key. In
            addition, it may contain a `count` key which contains the number
            of points in the bin.
        """
        value = d.get("mean", d.get("value", None), None)
        if value is None:
            raise ValueError("Dictionary must contain a mean or value key.")
        return cls(value=value, count=d.get("count", 1))

    def __getitem__(self, index):
        """Alternative method for getting the bin's mean and count.
        Parameters
        ----------
        index : int
            The index must be either 0 or 1, where 0 gets the mean (value),
            and 1 gets the count.
        """
        if index == 0:
            return self.value
        elif index == 1:
            return self.count
        raise IndexError("Invalid index (must be 0 or 1).")

    def __repr__(self):
        """Simple representation of a histogram bin.
        Returns
        -------
        Bin(value=`value`, count=`count`) where value and count are the bin's
        stored mean and count.
        """
        return "Bin(value=%d, count=%d)" % (self.value, self.count)

    def __iter__(self):
        """Iterator over the mean and count of this bin."""
        yield ("mean", self.value)
        yield ("count", self.count)

    def __str__(self):
        """String representation of a histogram bin."""
        return str(dict(self))

    def __eq__(self, obj):
        """Tests for equality of two bins.
        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value == obj.value

    def __lt__(self, obj):
        """Tests if this bin has a lower mean than another bin.
        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value < obj.value

    def __gt__(self, obj):
        """Tests if this bin has a higher mean than another bin.
        Parameters
        ----------
        obj : Bin
            The bin to which this bin's mean is compared.
        """
        return self.value > obj.value

    def __add__(self, obj):
        """Merge this bin with another bin and return the result.
        This method implements Step 7 from Algorithm 1 (Update) in ref [1].
        Parameters
        ----------
        obj : Bin
            The bin that will be merged with this bin.
        """
        count = float(self.count + obj.count)  # Summed heights
        if count:
            # Weighted average
            value = self.value * float(self.count) + obj.value * float(obj.count)
            value /= count
        else:
            value = 0.0
        return Bin(value=value, count=int(count))

    def __iadd__(self, obj):
        """Merge another bin into this one.
        Parameters
        ----------
        obj : Bin
            The bin that will be merged into this bin.
        """
        out = self + obj
        self = out
        return self


##==========================={Initialize statistics}============================


def init_stats_welford():
    p_PosX = RunningStats()
    p_PosY = RunningStats()
    p_PosZ = RunningStats()
    p_dPosX = RunningStats()
    p_dPosY = RunningStats()
    p_dPosZ = RunningStats()
    p_RotX = RunningStats()
    p_RotY = RunningStats()
    p_RotZ = RunningStats()
    p_dRotX = RunningStats()
    p_dRotY = RunningStats()
    p_dRotZ = RunningStats()
    p_dt = RunningStats()

    n_PosX = RunningStats()
    n_PosY = RunningStats()
    n_PosZ = RunningStats()
    n_dPosX = RunningStats()
    n_dPosY = RunningStats()
    n_dPosZ = RunningStats()
    n_RotX = RunningStats()
    n_RotY = RunningStats()
    n_RotZ = RunningStats()
    n_dRotX = RunningStats()
    n_dRotY = RunningStats()
    n_dRotZ = RunningStats()
    n_dT = RunningStats()

    features_dict = {
        "p_PosX": p_PosX,
        "p_PosY": p_PosY,
        "p_PosZ": p_PosZ,
        "p_dPosX": p_dPosX,
        "p_dPosY": p_dPosY,
        "p_dPosZ": p_dPosZ,
        "p_RotX": p_RotX,
        "p_RotY": p_RotY,
        "p_RotZ": p_RotZ,
        "p_dRotX": p_dRotX,
        "p_dRotY": p_dRotY,
        "p_dRotZ": p_dRotZ,
        "p_dt": p_dt,
        "n_PosX": n_PosX,
        "n_PosY": n_PosY,
        "n_PosZ": n_PosZ,
        "n_dPosX": n_dPosX,
        "n_dPosY": n_dPosY,
        "n_dPosZ": n_dPosZ,
        "n_RotX": n_RotX,
        "n_RotY": n_RotY,
        "n_RotZ": n_RotZ,
        "n_dRotX": n_dRotX,
        "n_dRotY": n_dRotY,
        "n_dRotZ": n_dRotZ,
        "n_dT": n_dT,
    }

    Welford_Struct = Struct(**features_dict)
    return Welford_Struct


def init_stats_tdigest():
    tp_PosX = TDigest()
    tp_PosY = TDigest()
    tp_PosZ = TDigest()
    tp_dPosX = TDigest()
    tp_dPosY = TDigest()
    tp_dPosZ = TDigest()
    tp_RotX = TDigest()
    tp_RotY = TDigest()
    tp_RotZ = TDigest()
    tp_dRotX = TDigest()
    tp_dRotY = TDigest()
    tp_dRotZ = TDigest()
    tp_dt = TDigest()

    tn_PosX = TDigest()
    tn_PosY = TDigest()
    tn_PosZ = TDigest()
    tn_dPosX = TDigest()
    tn_dPosY = TDigest()
    tn_dPosZ = TDigest()
    tn_RotX = TDigest()
    tn_RotY = TDigest()
    tn_RotZ = TDigest()
    tn_dRotX = TDigest()
    tn_dRotY = TDigest()
    tn_dRotZ = TDigest()
    tn_dT = TDigest()

    features_dict = {
        "tp_PosX": tp_PosX,
        "tp_PosY": tp_PosY,
        "tp_PosZ": tp_PosZ,
        "tp_dPosX": tp_dPosX,
        "tp_dPosY": tp_dPosY,
        "tp_dPosZ": tp_dPosZ,
        "tp_RotX": tp_RotX,
        "tp_RotY": tp_RotY,
        "tp_RotZ": tp_RotZ,
        "tp_dRotX": tp_dRotX,
        "tp_dRotY": tp_dRotY,
        "tp_dRotZ": tp_dRotZ,
        "tp_dt": tp_dt,
        "tn_PosX": tn_PosX,
        "tn_PosY": tn_PosY,
        "tn_PosZ": tn_PosZ,
        "tn_dPosX": tn_dPosX,
        "tn_dPosY": tn_dPosY,
        "tn_dPosZ": tn_dPosZ,
        "tn_RotX": tn_RotX,
        "tn_RotY": tn_RotY,
        "tn_RotZ": tn_RotZ,
        "tn_dRotX": tn_dRotX,
        "tn_dRotY": tn_dRotY,
        "tn_dRotZ": tn_dRotZ,
        "tn_dT": tn_dT,
    }
    td_Struct = Struct(**features_dict)

    return td_Struct


##==========================={Feed data for streaming-metrics}==================

# def feed_stream(stream, data, weight= 1, col=None, val=0, flag=None):
#    if type(stream) == TDigest:
#        if type(data) == pd.core.frame.DataFrame:
#            assert col is not None, 'Provide column for pandas dataframe or data as a numpy ndarray'
#            [stream.push(x, weight) for x in get_subset(data, col, val, flag)[col] ]
#        elif type(data) == np.ndarray:
#            [stream.push(x, weight) for x in get_subset(data, val, flag)]
#        else:
#            raise TypeError('Provide a pandas dataframe or numpy array instead of {}'.format(type(data)))
#    if type(stream) == RunningStats:
#        if type(data) == pd.core.frame.DataFrame:
#            assert col is not None, 'Provide column for pandas dataframe or data as a numpy ndarray'
#            assert data[col] is not None, 'No Data'
#            [stream.push(x) for x in get_subset(data, col, val, flag)[col]]
#        elif type(data) == np.ndarray:
#            [stream.push(x) for x in get_subset(data, val, flag)]
#        else:
#            raise TypeError('Provide a pandas dataframe or numpy array instead of {}'.format(type(data)))


import ctypes as _ctypes
import sys as _sys
from sys import platform as _platform
from math import log, sqrt
import types
import collections
from numbers import Number as numeric_types

iterator_types = (types.GeneratorType, collections.Iterable)

if _sys.version_info.major >= 3:
    _izip = zip
else:
    from itertools import izip as _izip

try:
    from itertools import accumulate
except ImportError:
    # itertools.accumulate only in Py3.x
    def accumulate(iterable):
        it = iter(iterable)
        total = next(it)
        yield total
        for element in it:
            total += element
            yield total


_E = 2.718281828459045

__all__ = ["next_after", "bin_sums", "argmin", "bin_diff", "accumulate"]

if _platform == "linux" or _platform == "linux2":
    _libm = _ctypes.cdll.LoadLibrary("libm.so.6")
    _funcname = "nextafter"
elif _platform == "darwin":
    _libm = _ctypes.cdll.LoadLibrary("libSystem.dylib")
    _funcname = "nextafter"
elif _platform == "win32":
    _libm = _ctypes.cdll.LoadLibrary("msvcrt.dll")
    _funcname = "_nextafter"
else:
    # these are the ones I have access to...
    # fill in library and function name for your system math dll
    print("Platform", repr(_platform), "is not supported")
    _sys.exit(0)

_nextafter = getattr(_libm, _funcname)
_nextafter.restype = _ctypes.c_double
_nextafter.argtypes = [_ctypes.c_double, _ctypes.c_double]


def next_after(x, y):
    """Returns the next floating-point number after x in the direction of y."""
    # This implementation comes from here:
    # http://stackoverflow.com/a/6163157/1256988
    return _nextafter(x, y)


def _diff(a, b, weighted):
    diff = b.value - a.value
    if weighted:
        diff *= log(_E + min(a.count, b.count))
    return diff


def bin_diff(array, weighted=False):
    return [_diff(a, b, weighted) for a, b in _izip(array[:-1], array[1:])]


def argmin(array):
    # Turns out Python's min and max functions are super fast!
    # http://lemire.me/blog/archives/2008/12/17/fast-argmax-in-python/
    return array.index(min(array))


def bin_sums(array, less=None):
    return [
        (a.count + b.count) / 2.
        for a, b in _izip(array[:-1], array[1:])
        if less is None or b.value <= less
    ]


def linspace(start, stop, num):
    """Custom version of numpy's linspace to avoid numpy depenency."""
    if num == 1:
        return stop
    h = (stop - start) / float(num)
    values = [start + h * i for i in range(num + 1)]
    return values


def roots(a, b, c):
    """Super simple quadratic solver."""
    d = b ** 2.0 - (4.0 * a * c)
    if d < 0:
        raise (ValueError("This equation has no real solution!"))
    elif d == 0:
        x = (-b + sqrt(d)) / (2.0 * a)
        return (x, x)
    else:
        x1 = (-b + sqrt(d)) / (2.0 * a)
        x2 = (-b - sqrt(d)) / (2.0 * a)
        return (x1, x2)
