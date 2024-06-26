import numpy as np
from numpy import ma
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import Formatter, FixedLocator
from scipy.optimize import curve_fit


def logistic(x, L, k, x0):
    """Logistic function (s-curve)."""
    return L / (1 + np.exp(-k * (x - x0)))


class ProbabilityScale(mscale.ScaleBase):
    """
    Scales data so that points along a logistic curve become evenly spaced.
    """

    # The scale class must have a member ``name`` that defines the
    # string used to select the scale.  For example,
    # ``gca().set_yscale("probability")`` would be used to select this
    # scale.
    name = "probability"

    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.

        lower_bound: Minimum value of x. Defaults to .01.
        upper_bound_dist: L - upper_bound_dist is the maximum value
        of x. Defaults to lower_bound.

        """
        mscale.ScaleBase.__init__(self)
        lower_bound = kwargs.pop("lower_bound", .01)
        if lower_bound <= 0:
            raise ValueError("lower_bound must be greater than 0")
        self.lower_bound = lower_bound
        upper_bound_dist = kwargs.pop("upper_bound_dist", lower_bound)
        self.points = kwargs["points"]
        # determine parameters of logistic function with curve fitting
        x = np.linspace(0, 1, len(self.points))
        # initial guess for parameters
        p0 = [max(self.points), 1, .5]
        popt, pcov = curve_fit(logistic, x, self.points, p0=p0)
        [self.L, self.k, self.x0] = popt
        self.upper_bound = self.L - upper_bound_dist

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The ProbabilityTransform class is defined below as a
        nested class of this one.
        """
        return self.ProbabilityTransform(
            self.lower_bound, self.upper_bound, self.L, self.k, self.x0
        )

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters is rather outside the scope of this example, but
        there are many helpful examples in ``ticker.py``.
        """

        axis.set_major_locator(FixedLocator(self.points))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In this case, the bounds should be
        limited to the threshold that was passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(vmin, self.lower_bound), min(vmax, self.upper_bound)

    class ProbabilityTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, lower_bound, upper_bound, L, k, x0):
            ''''''
            mtransforms.Transform.__init__(self)
            self.lower_bound = lower_bound
            self.L = L
            self.k = k
            self.x0 = x0
            self.upper_bound = upper_bound

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the scale
            is limited by the user-specified threshold, the input
            array must be masked to contain only valid values.
            ``matplotlib`` will handle masked arrays and remove the
            out-of-range data from the plot.  Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """
            masked = ma.masked_where((a < self.lower_bound) | (a > self.upper_bound), a)
            return ma.log((self.L - masked) / masked) / -self.k + self.x0

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return ProbabilityScale.InvertedProbabilityTransform(
                self.lower_bound, self.upper_bound, self.L, self.k, self.x0
            )

    class InvertedProbabilityTransform(mtransforms.Transform):
        ''''''
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, lower_bound, upper_bound, L, k, x0):
            ''''''
            mtransforms.Transform.__init__(self)
            self.lower_bound = lower_bound
            self.L = L
            self.k = k
            self.x0 = x0
            self.upper_bound = upper_bound

        def transform_non_affine(self, a):
            """
            * ---------------Function---------------
* Applies a non-affine transformation to the input value 'a'.
* 
* ----------------Returns---------------
* -> float: The transformed value
* 
* ----------------Params----------------
* a : float
    The input value to be transformed
* self.L : float
    A parameter of the transformation
* self.k : float
    A parameter of the transformation
* self.x0 : float
    A parameter of the transformation
* 
* ----------------Usage-----------------
* >>> transformed_value = transform_non_affine(1.5)
            """
            return self.L / (1 + np.exp(-self.k * (a - self.x0)))

        def inverted(self):
            '''
* ---------------Function---------------
* Creates an inverted probability transform object.
* 
* ----------------Returns---------------
* -> ProbabilityTransform: The inverted probability transform object
* 
* ----------------Params----------------
* None
* 
* ----------------Usage-----------------
* >>> inverted_transform = inverted()
* print(inverted_transform.lower_bound)
            '''
            return ProbabilityScale.ProbabilityTransform(
                self.lower_bound, self.upper_bound, self.L, self.k, self.x0
            )


# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(ProbabilityScale)


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     x = np.linspace(.1, 100, 1000)
#     points = np.array([.2,.5,1,2,5,10,20,30,40,50,60,70,80,90,95,98])

#     plt.plot(x, x)
#     plt.gca().set_xscale('probability', points = points, vmin = .01)
#     plt.grid(True)

#     plt.show()


import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import FormatStrFormatter, FixedLocator
from scipy.stats import norm


# TODO transform_non_affine -> use inverse of cdf (quantiles)
# to compress the middle and increase  tails resolution
class ProbScale(mscale.ScaleBase):
    """
    Scales data in range 0 to 100 using a non-standard log transform
    This scale attempts to replicate "probability paper" scaling

    The scale function:
        A piecewise combination of exponential, linear, and logarithmic scales

    The inverse scale function:
      piecewise combination of exponential, linear, and logarithmic scales

    Since probabilities at 0 and 100 are not represented,
    there is user-defined upper and lower limit, above and below which nothing
    will be plotted.  This defaults to .1 and 99 for lower and upper, respectively.

    """

    # The scale class must have a member ``name`` that defines the
    # string used to select the scale.  For example,
    # ``gca().set_yscale("mercator")`` would be used to select this
    # scale.
    name = "prob_scale"

    def __init__(self, axis, **kwargs):
        """
        Any keyword arguments passed to ``set_xscale`` and
        ``set_yscale`` will be passed along to the scale's
        constructor.

        upper: The probability above which to crop the data.
        lower: The probability below which to crop the data.
        """
        mscale.ScaleBase.__init__(self)
        upper = kwargs.pop("upper", 98)  # Default to an upper bound of 98%
        if upper <= 0 or upper >= 100:
            raise ValueError("upper must be between 0 and 100.")
        lower = kwargs.pop("lower", 0.2)  # Default to a lower bound of .2%
        if lower <= 0 or lower >= 100:
            raise ValueError("lower must be between 0 and 100.")
        if lower >= upper:
            raise ValueError("lower must be strictly less than upper!.")
        self.lower = lower
        self.upper = upper

        # This scale is best described by the CDF of the normal distribution
        # This distribution is paramaterized by mu and sigma, these default vaules
        # are provided to work generally well, but can be adjusted by the user if desired
        mu = kwargs.pop("mu", 15)
        sigma = kwargs.pop("sigma", 40)
        self.mu = mu
        self.sigma = sigma
        # Need to enfore the upper and lower limits on the axes initially
        axis.axes.set_xlim(lower, upper)

    def get_transform(self):
        """
        Override this method to return a new instance that does the
        actual transformation of the data.

        The ProbTransform class is defined below as a
        nested class of this one.
        """
        return self.ProbTransform(self.lower, self.upper, self.mu, self.sigma)

    def set_default_locators_and_formatters(self, axis):
        """
        Override to set up the locators and formatters to use with the
        scale.  This is only required if the scale requires custom
        locators and formatters.  Writing custom locators and
        formatters: many helpful examples in ``ticker.py``.

        In this case, the prob_scale uses a fixed locator from
        0.1 to 99 % and a custom no formatter class

        This builds both the major and minor locators, and cuts off any values
        above or below the user defined thresholds: upper, lower
        """
        # major_ticks = np.asarray([.2,.5,1,2,5,10,20,30,40,50,60,70,80,90,95,98])
        major_ticks = np.asarray(
            [.2, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 98]
        )  # removed a couple ticks to make it look nicer
        major_ticks = major_ticks[
            np.where((major_ticks >= self.lower) & (major_ticks <= self.upper))
        ]

        minor_ticks = np.concatenate(
            [
                np.arange(.2, 1, .1),
                np.arange(1, 2, .2),
                np.arange(2, 20, 1),
                np.arange(20, 80, 2),
                np.arange(80, 98, 1),
            ]
        )
        minor_ticks = minor_ticks[
            np.where((minor_ticks >= self.lower) & (minor_ticks <= self.upper))
        ]
        axis.set_major_locator(FixedLocator(major_ticks))
        axis.set_minor_locator(FixedLocator(minor_ticks))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Override to limit the bounds of the axis to the domain of the
        transform.  In the case of Probability, the bounds should be
        limited to the user bounds that were passed in.  Unlike the
        autoscaling provided by the tick locators, this range limiting
        will always be adhered to, whether the axis range is set
        manually, determined automatically or changed through panning
        and zooming.
        """
        return max(self.lower, vmin), min(self.upper, vmax)

    class ProbTransform(mtransforms.Transform):
        # There are two value members that must be defined.
        # ``input_dims`` and ``output_dims`` specify number of input
        # dimensions and output dimensions to the transformation.
        # These are used by the transformation framework to do some
        # error checking and prevent incompatible transformations from
        # being connected together.  When defining transforms for a
        # scale, which are, by definition, separable and have only one
        # dimension, these members should always be set to 1.
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, upper, lower, mu, sigma):
            ''''''
            mtransforms.Transform.__init__(self)
            self.upper = upper
            self.lower = lower
            self.mu = mu
            self.sigma = sigma

        def transform_non_affine(self, a):
            """
            This transform takes an Nx1 ``numpy`` array and returns a
            transformed copy.  Since the range of the Probability scale
            is limited by the user-specified threshold, the input
            array must be masked to contain only valid values.
            ``matplotlib`` will handle masked arrays and remove the
            out-of-range data from the plot.  Importantly, the
            ``transform`` method *must* return an array that is the
            same shape as the input array, since these values need to
            remain synchronized with values in the other dimension.
            """

            masked = np.ma.masked_where((a < self.upper) & (a > self.lower), a)
            # Get the CDF of the normal distribution located at mu and scaled by sigma
            # Multiply these by 100 to put it into a percent scale
            cdf = norm.cdf(masked, self.mu, self.sigma) * 100
            return cdf

        def inverted(self):
            """
            Override this method so matplotlib knows how to get the
            inverse transform for this transform.
            """
            return ProbScale.InvertedProbTransform(
                self.lower, self.upper, self.mu, self.sigma
            )

    class InvertedProbTransform(mtransforms.Transform):
        '''
* ------------InvertedProbTransform---------------
* A probability transform that inverts the probability scale.
* ----------------Params----------------
* lower :: float
*   The lower bound of the transformed range.
* upper :: float
*   The upper bound of the transformed range.
* mu :: float
*   The mean of the normal distribution.
* sigma :: float
*   The standard deviation of the normal distribution.
* ----------------Methods----------------
* 
* ---transform_non_affine---------------
* 
* Transform a value `a` from the probability range [0,1] to the transformed range [lower, upper].
* 
* Parameters:
* 
* a :: float
*   The value to transform, in the probability range [0,1].
* 
* Returns:
* 
* inverse :: float
*   The transformed value in the range [lower, upper].
* 
* ---inverted---------------
* 
* Return an instance of `ProbScale.ProbTransform` that inverts the transformation.
* 
* Returns:
* 
* prob_transform :: ProbScale.ProbTransform
*   An instance of `ProbScale.ProbTransform` that inverts the transformation.
        '''
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self, lower, upper, mu, sigma):
            """"""
            mtransforms.Transform.__init__(self)
            self.lower = lower
            self.upper = upper
            self.mu = mu
            self.sigma = sigma

        def transform_non_affine(self, a):
            """
            * ---------------Function---------------
* Applies the probability transform to a given value
* ----------------Returns---------------
* -> float | The transformed value
* ----------------Params----------------
* a :: float | The value to transform, in percent scale [0,100]
* ----------------Usage-----------------
* Apply the probability transform to a value in percent scale [0,100].
            """
            # Need to get the PPF value for a, which is in a percent scale [0,100], so move back to probability range [0,1]
            inverse = norm.ppf(a / 100, self.mu, self.sigma)
            return inverse

        def inverted(self):
            '''
            * ---------------Function---------------
* Returns a new probability transform object with the same bounds but inverted
* ----------------Returns---------------
* -> ProbScale.ProbTransform | A new probability transform object
* ----------------Params----------------
* None
* ----------------Usage-----------------
* Get a new probability transform object with the same bounds but inverted.
            '''
            return ProbScale.ProbTransform(self.lower, self.upper)


# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(ProbScale)


# axes.set_ylabel('Discharge in CFS')
# axes.set_xlabel('Exceedance Probability')
# plt.setp(plt.xticks()[1], rotation=45)
# #Adjust the scales of the x and y axis
# axes.set_yscale('log', basey=10, subsy=[2,3,4,5,6,7,8,9])
# axes.set_xscale('prob_scale', upper=98, lower=.2)
# #Adjust the yaxis labels and format
# axes.yaxis.set_minor_locator(FixedLocator([200, 500, 1500, 2500, 3500, 4500, 5000, 6000, 7000, 8000, 9000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]))
# axes.yaxis.set_minor_formatter(FormatStrFormatter('%d'))
# axes.yaxis.set_major_formatter(FormatStrFormatter('%d'))

# #Finally set the y-limit of the plot to be reasonable
# axes.set_ylim((0, 2*pp['Q'].max()))
# #Invert the x-axis
# axes.invert_xaxis()
# #Turn on major and minor grid lines
# axes.grid(which='both', alpha=.9)
