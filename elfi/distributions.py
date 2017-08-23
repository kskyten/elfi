import numpy as np
import scipy.stats as ss
from scipy.interpolate import interp1d


__all__ = ('EmpiricalDensity', 'ecdf', 'eppf', 'empirical_densities', 'MetaGaussian')


def ecdf(samples):
    """Compute an empirical cdf.

    Parameters
    ----------
    samples : array_like
      a univariate sample

    Returns
    -------
    empirical_cdf
      an interpolated function for the estimated cdf
    """
    x, y = _handle_endpoints(*_ecdf(samples))
    return _interp_ecdf(x, y)


def eppf(samples):
    """Compute an empirical quantile function.

    Parameters
    ----------
    samples : array_like
      a univariate sample

    Returns
    -------
    empirical_ppf
      an interpolated function for the estimated quantile function
    """
    x, y = _handle_endpoints(*_ecdf(samples))
    return _interp_ppf(x, y)


def _ecdf(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) +  1)/float(len(xs))
    return xs, ys


def _low(xs, ys):
    """Compute the intercetion point (x, 0)."""
    slope = (ys[1] - ys[0])/(xs[1] - xs[0])
    intersect = -ys[0]/slope + xs[0]
    return intersect


def _high(xs, ys):
    """Compute the interception point (x, 1)."""
    slope = (ys[-1] - ys[-2])/(xs[-1] - xs[-2])
    intersect = (1 - ys[-1])/slope + xs[-1]
    return intersect


def _handle_endpoints(xs, ys):
    high = _high(xs, ys)
    low = _low(xs, ys)

    # add endpoints
    x = np.append(np.insert(xs, 0, low), high)
    y = np.append(np.insert(ys, 0, 0.), 1.)
    return x, y


def _interp_ecdf(x, y, **kwargs):
    low, high = x[0], x[-1]

    # linear interpolation
    f = interp1d(x, y, **kwargs)

    def interp(q):
        if isinstance(q, np.ndarray):
            too_low = sum(q < low)
            too_high = sum(q > high)
            return np.concatenate([np.zeros(too_low),
                                   f(q[(q >= low) & (q <= high)]),
                                   np.ones(too_high)])
        else:
            return _scalar_cdf(f, low, high, q)

    return interp


def _scalar_cdf(f, low, high, q):
    if q < low:
        return 0
    elif q > high:
        return 1
    else:
        return f(q)


def _interp_ppf(x, y, **kwargs):
    f = interp1d(y, x, **kwargs)

    def interp(p):
        try:
            return f(p)
        except ValueError:
            raise ValueError("The quantile function is not defined outside [0, 1].")

    return interp


class EmpiricalDensity(object):
    """An empirical approximation of a random variable.

    The density function is approximated using the gaussian
    kernel density estimation from scipy (scipy.stats.gaussian_kde).
    The cumulative distribution function and quantile function are constructed
    the linearly interpolated empirical cumulative distribution function.

    Parameters
    ----------
    samples : np.ndarray
        a univariate sample
    **kwargs
        additional arguments for kernel density estimation

    Attributes
    ----------
    kde :
        a Gaussian kernel density estimate
    """

    def __init__(self, samples, **kwargs):
        self.kde = ss.gaussian_kde(samples, **kwargs)
        x, y = _handle_endpoints(*_ecdf(samples))
        self.cdf = _interp_ecdf(x, y)
        self.ppf = _interp_ppf(x, y)

    @property
    def dataset(self):
        """The dataset used for fitting the kernel density estimate."""
        return self.kde.dataset

    @property
    def n(self):
        """The number of samples used for the kernel density estimation."""
        return self.kde.n

    def pdf(self, x):
        """Compute the estimated pdf."""
        return self.kde.pdf(x)

    def logpdf(self, x):
        """Compute the estimated logarithmic pdf."""
        return self.kde.logpdf(x)

    def rvs(self, n):
        """Sample n values from the empirical density."""
        return self.ppf(np.random.rand(n))


def estimate_densities(marginal_samples, **kwargs):
    """Compute Gaussian kernel density estimates.

    Parameters
    ----------
    marginal_samples : np.ndarray
        a NxM array of N observations in M variables
    **kwargs :
        additional arguments for kernel density estimation

    Returns
    -------
    empirical_densities :
       a list of EmpiricalDensity objects
    """
    return [EmpiricalDensity(marginal_samples[:, i], **kwargs)
            for i in range(marginal_samples.shape[1])]



def _raise(err):
    """Exception raising closure."""
    def fun():
        raise err
    return fun


class MetaGaussian(object):
    """A meta-Gaussian distribution

    Parameters
    ----------
    corr : np.ndarray
        Th correlation matrix of the meta-Gaussian distribution.
    marginals : density_like
        A list of objects that implement 'cdf' and 'ppf' methods.
    marginal_samples : np.ndarray
        A NxM array of samples, where N is the number of observations
        and m is the number of dimensions.

    Attributes
    ----------
    corr : np.ndarray
        Th correlation matrix of the meta-Gaussian distribution.
    marginals : List
        a list of marginal densities

    References
    ----------
    Jingjing Li, David J. Nott, Yanan Fan, Scott A. Sisson (2016)
    Extending approximate Bayesian computation methods to high dimensions
    via Gaussian copula.
    https://arxiv.org/abs/1504.04093v1
    """

    def __init__(self, corr, marginals=None, marginal_samples=None):
        self._handle_marginals(marginals, marginal_samples)
        self.corr = corr

    def _handle_marginals(self, marginals, marginal_samples):
        marginalp = marginals is not None
        marginal_samplesp = marginal_samples is not None
        {(False, False): _raise(ValueError("Must provide either marginals or marginal_samples.")),
         (True, False): self._handle_marginals1,
         (False, True): self._handle_marginals2,
         (True, True): self._handle_marginals3}.get((marginalp, marginal_samplesp))(marginals,
                                                                                    marginal_samples)

    def _handle_marginals1(self, marginals, marginal_samples):
        self.marginals = marginals

    def _handle_marginals2(self, marginals, marginal_samples):
        self.marginals = estimate_densities(marginal_samples)

    def _handle_marginals3(self, marginals, marginal_samples):
        self.marginals = marginals
        # TODO: Maybe notify that marginal_samples is not used?

    def logpdf(self, theta):
        """Evaluate the logarithm of the density function of the meta-Gaussian distribution.

        Parameters
        ----------
        theta : np.ndarray
            the evaluation point

        See Also
        --------
        pdf
        """
        if len(theta.shape) == 1:
            return self._logpdf(theta)
        elif len(theta.shape) == 2:
            return np.array([self._logpdf(t) for t in theta])

    def pdf(self, theta):
        r"""Evaluate the probability density function of the meta-Gaussian distribution.

        The probability density function is given by

        .. math::
            g(\theta) = \frac{1}{|R|^{\frac12}} \exp \left\{-\frac{1}{2} u^T (R^{-1} - I) u \right\}
                        \prod_{i=1}^p g_i(\theta_i) \, ,

        where :math:`\phi` is the standard normal density, :math:`u_i = \Phi^{-1}(G_i(\theta_i))`,
        :math:`g_i` are the marginal densities, and :math:`R` is a correlation matrix.

        Parameters
        ----------
        theta : np.ndarray
            the evaluation point

        See Also
        --------
        logpdf
        """
        return np.exp(self.logpdf(theta))

    def _marginal_prod(self, theta):
        """Evaluate the logarithm of the product of the marginals."""
        res = 0
        for (i, t) in enumerate(theta):
            res += self.marginals[i].logpdf(t)
        return res

    def _eta_i(self, i, t):
        return ss.norm.ppf(self.marginals[i].cdf(t))

    def _eta(self, theta):
        return np.array([self._eta_i(i, t) for (i, t) in enumerate(theta)])

    def _logpdf(self, theta):
        correlation_matrix = self.corr
        n, m = correlation_matrix.shape
        a = np.log(1/np.sqrt(np.linalg.det(correlation_matrix)))
        L = np.eye(n) - np.linalg.inv(correlation_matrix)
        quadratic = 1/2 * self._eta(theta).T.dot(L).dot(self._eta(theta))
        c = self._marginal_prod(theta)
        return a + quadratic + c

    def _plot_marginal(self, inx, bounds, points=100):
        import matplotlib.pyplot as plt
        t = np.linspace(*bounds, points)
        return plt.plot(t, self.marginals[inx].pdf(t))

    __call__ = logpdf
