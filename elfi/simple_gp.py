import numpy as np
import matplotlib.pyplot as plt


class SimpleGP(object):
    """
    A naive implementation of a Gaussian Process for testing purposes.
    """

    def __init__(self, input_dim, kernel=None, bounds=None, mean_fun=None):
        self.input_dim = input_dim
        if bounds is not None:
            self.bounds = bounds
        else:
            print("GpyModel: No bounds supplied, defaulting to [0,1] bounds.")
            self.bounds = [(0,1)] * self.input_dim
        if kernel is None:
            self.cov_fun = _cov_fun
        else:
            self.cov_fun = kernel
        if mean_fun is None:
            self.mean_fun = _mean_fun
        else:
            self.mean_fun = mean_fun
        self.X = np.empty((0, input_dim))
        self.Y = np.empty((0, 1))

    def update(self, X, Y):
        """
            Add (X, Y) as observations, updates GP model.
            X and Y should be 2d numpy arrays with observations in rows.
        """
        self.X = np.vstack((self.X, X))
        self.Y = np.vstack((self.Y, Y))

        means = self.mean_fun(self.X)
        covariances = self.cov_fun(self.X[:, None, :], self.X[None, :, :])
        self._L = np.linalg.cholesky(covariances)
        self._alpha = np.linalg.solve( self._L.T,
            np.linalg.solve( self._L, self.Y - means )
                               )
    def n_observations(self):
        """ Returns the number of observed samples """
        return self.Y.shape[0]

    def evaluate(self, x):
        """ Returns the mean, variance and std of the GP at x as floats """
        x2d = np.atleast_2d(x)
        k_t = self.cov_fun(self.X[:, None, :], x2d[None, :, :])
        mu_t = self.mean_fun(x) + k_t.T.dot(self._alpha)
        v = np.linalg.solve(self._L, k_t)
        var_t = self.cov_fun(x2d[None, :, :], x2d[None, :, :]) - v.T.dot(v)
        return float(mu_t), float(var_t), np.sqrt(float(var_t))

    def eval_mean(self, x):
        m, s2, s = self.evaluate(x)
        return m

    def plot(self, fun=None, n_xaxis=50):
        """
        Plot mean and covariance of GP.

        Parameters
        ----------
        fun: function
            The modeled function.
        n_xaxis: int
            Number of points on x-axis to evaluate.

        Returns
        -------
        fig: matplotlib.figure.Figure
        ax: numpy.array(matplotlib.axes._subplots.AxesSubplot)
        """
        fig, ax = plt.subplots(nrows=self.input_dim)
        ax = np.atleast_1d(ax)
        for ii in range(self.input_dim):
            xs = np.linspace(*self.bounds[ii], n_xaxis)
            mus = np.empty(n_xaxis)
            sigmas = np.empty(n_xaxis)
            for jj in range(n_xaxis):
                mus[jj], _, sigmas[jj] = self.evaluate(xs[jj])
            if fun is not None:
                ax[ii].plot(xs, fun(xs), '--g', lw=3);
            ax[ii].plot(xs, mus, '-b', lw=3);
            ax[ii].fill_between(xs, mus-1.96*sigmas, mus+1.96*sigmas, alpha=0.5);
            ax[ii].plot(self.X[:, ii], self.Y[:, 0], 'ro', markersize=10);
        return fig, ax


def _mean_fun(x):
    """
    Mean function for GP.
    """
    return 0


def _cov_fun(x1, x2, gamma=1.):
    """
    Covariance function for GP.
    """
    return np.exp( gamma * -np.sum((x1 - x2)**2., axis=2))
