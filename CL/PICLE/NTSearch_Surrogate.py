import GPy
from GPy.kern import Kern
from GPy.core.parameterization import Param
from GPyOpt.util.general import get_quantiles
from paramz.transformations import Logexp
import numpy as np
from typing import List, Union, Optional
from GPy.util.normalizer import Standardize


class CustomMeanNormalizer(Standardize):
    def scale_by(self, Y):
        Y = np.ma.masked_invalid(Y, copy=False)
        self.mean = Y.mean(0).view(np.ndarray)
        self.std = np.ones_like(Y.std(0).view(np.ndarray))


class CustomRBFKernel(Kern):
    """ Changed so that I understand better the functions involved."""
    def __init__(self, full_distances: np.ndarray, variance: float = 1., lengthscale: float=1.):
        super(CustomRBFKernel, self).__init__(1, None, 'custom rbf kernel')
        self.full_distances = full_distances

        lengthscale = np.asarray(lengthscale)
        self.lengthscale = Param('lengthscale', lengthscale, Logexp())
        self.variance = Param('variance', np.ones(1)*variance, Logexp())

        assert lengthscale.size == 1, "Only 1 lengthscale needed for non-ARD kernel"
        assert self.variance.size==1

        self.link_parameters(self.variance, self.lengthscale)

    def parameters_changed(self):
        # nothing to do here
        pass

    def _unscaled_dist(self, X, X2):
        if X2 is None:
            X2 = X

        X = X.astype(int).reshape((-1,))
        X2 = X2.astype(int).reshape((-1,))
        returning_kernel = self.full_distances[np.ix_(X, X2)]
        np.fill_diagonal(returning_kernel, 0)
        return returning_kernel

    def _scaled_dist(self, X, X2=None):
        """
        Efficiently compute the scaled distance, r.

        ..math::
            r = \sqrt( \sum_{q=1}^Q (x_q - x'q)^2/l_q^2 )

        Note that if there is only one lengthscale, l comes outside the sum. In
        this case we compute the unscaled distance first (in a separate
        function for caching) and divide by lengthscale afterwards
        """
        return self._unscaled_dist(X, X2)/self.lengthscale

    def K(self, X, X2=None):
        """
        Kernel function applied on inputs X and X2.
        In the stationary case there is an inner function depending on the
        distances from X to X2, called r.

        K(X, X2) = K_of_r((X-X2)**2)
        """
        r = self._scaled_dist(X, X2)
        return self.K_of_r(r)

    def K_of_r(self, r):
        return self.variance * np.exp(-0.5 * r ** 2)

    def Kdiag(self, X):
        ret = np.empty(X.shape[0])
        ret[:] = self.variance
        return ret

    def dK_dr_via_X(self, X, X2):
        """compute the derivative of K wrt X going through X"""
        # a convenience function, so we can cache dK_dr
        return self.dK_dr(self._scaled_dist(X, X2))

    def dK_dr(self, r):
        return -r * self.K_of_r(r)

    def update_gradients_full(self, dL_dK, X, X2):
        """
        Given the derivative of the objective wrt the covariance matrix
        (dL_dK), compute the gradient wrt the parameters of this kernel,
        and store in the parameters object as e.g. self.variance.gradient
        """
        self.variance.gradient = np.sum(self.K(X, X2) * dL_dK) / self.variance

        # now the lengthscale gradient(s)
        dL_dr = self.dK_dr_via_X(X, X2) * dL_dK
        r = self._scaled_dist(X, X2)
        self.lengthscale.gradient = -np.sum(dL_dr * r) / self.lengthscale

    def reset_gradients(self):
        self.variance.gradient = 0.
        self.lengthscale.gradient = 0.


class GPModel:
    def __init__(self, full_distances_matrix: np.ndarray):
        self.full_distances_matrix = full_distances_matrix

        self.gp = None
        self.fmin = None

        self.threshold = 0.005

    def clear(self):
        pass

    def update_model(self, X_tr, Y_tr):
        kernel = CustomRBFKernel(self.full_distances_matrix)
        c_mean_normalizer = CustomMeanNormalizer()
        self.gp = GPy.models.GPRegression(X_tr, Y_tr, kernel, normalizer=c_mean_normalizer, noise_var=0.01)
        self.gp.optimize(messages=False)
        self.fmin = np.min(Y_tr)

    def get_aq_EI(self, pred_mean, pred_std):
        """
        fmin = the current smallest f value observed
        """
        jitter = 0.001
        phi, Phi, u = get_quantiles(jitter, self.fmin, pred_mean, pred_std)
        f_acqu = pred_std * (u * Phi + phi)
        return f_acqu

    def select_next_index(self, X_rest):
        ucb_lcb_beta = 2.

        c_predictions = self.gp.predict(X_rest)
        predicted_variances = c_predictions[1]
        predicted_variances[predicted_variances < 0] = 0.  # fix numerical inaccuracies
        predicted_stds = np.sqrt(predicted_variances)

        aq_EI = self.get_aq_EI(c_predictions[0], predicted_stds)
        aq_LCB = c_predictions[0] - ucb_lcb_beta * predicted_stds  # predicted_variances

        argmin_lcb = np.argmin(aq_LCB, axis=0).item()
        # argmin_ucb = np.argmax(aq_EI, axis=0).item()

        aq_min_ei = aq_EI[argmin_lcb].item()
        aq_min_LCB = aq_LCB[argmin_lcb].item()

        return argmin_lcb, aq_min_ei, aq_min_LCB