from collections import namedtuple
import numpy as np
import numpy.linalg
from sklearn import random_projection


def _stack_numpy_list(np_list):
    dss = np_list.shape
    # if it's a list, stack it on top, as we assume the list items are of the same type
    return np_list.reshape(-1, dss[2], dss[3], dss[4])


def get_log_pdf(data: np.ndarray, mean: np.ndarray, cov: np.ndarray):
    n = mean.size

    p1 = 0.5 * n * np.log(2*np.pi)

    log_sign_cov_det, log_cov_det = np.linalg.slogdet(cov)
    assert log_sign_cov_det > 0
    p2 = 0.5 * log_cov_det

    cov_inv = np.linalg.inv(cov)
    data_minus_mean = data - mean
    p3_a = cov_inv @ data_minus_mean.T
    p3_b = (data_minus_mean * p3_a.T).sum(axis=1)
    p3 = 0.5 * p3_b

    log_pdf = -p1 -p2 -p3
    return log_pdf


class SoftType:
    # the approximated input distribution of a module is referred to as an input soft type
    def __init__(self, mean: np.array, cov: np.array, random_projector=None):
        self.mean = mean
        self.cov = cov
        self.random_projector = random_projector

    def get_nll_of_processed_datapoints(self, processed_data_points):
        data_points = processed_data_points
        if self.random_projector is not None:
            data_points = self.random_projector.fit_transform(data_points)

        log_pdfs = get_log_pdf(data_points, self.mean, self.cov)
        nll = -log_pdfs.sum()
        return nll


    @staticmethod
    def process_data_points(data_points: np.array):
        """
        :param data_points: a numpy array of size [num_data_points, num_features]
        """
        if type(data_points) == list:
            data_points = np.vstack(data_points)
        else:
            assert len(data_points.shape) > 1

            # check if its a list of images
            if len(data_points.shape) == 5:
                # if it's a list, stack it on top, as we assume the list items are of the same type
                data_points = _stack_numpy_list(data_points)

        if len(data_points.shape) > 2:
            # if necessary, reshape to a 2d tensor
            data_points = data_points.reshape(data_points.shape[0], -1)
        return data_points

    @staticmethod
    def create_from_data_points(data_points: np.array, use_random_projection=False, target_dim=None, rndm_proj_random_state=27):
        """
        :param data_points: a numpy array of size [num_data_points, num_features]
        :param use_random_projection
        :param target_dim:  will return a [1xtarget_dim] vector
        :return:
        """
        data_points = SoftType.process_data_points(data_points)

        transformer = None
        if use_random_projection:
            assert target_dim is not None
            if data_points.shape[1] > target_dim:
                transformer = random_projection.GaussianRandomProjection(n_components=target_dim, random_state=rndm_proj_random_state)
                data_points = transformer.fit_transform(data_points)

        st_mean = np.mean(data_points, axis=0)
        st_cov = np.cov(data_points, rowvar=False, ddof=1)

        return SoftType(st_mean, st_cov, transformer)
