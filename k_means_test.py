import k_means
import numpy as np
import sklearn.cluster
import numpy.testing
from sklearn.utils.testing import assert_raise_message
from sklearn.utils import check_random_state


def no_points():
    X = np.asarray([])
    assert_raise_message(ValueError, "The input array should not contain any singleton dimensions",
                         k_means.k_means, X, 2)


def one_point():
    X = np.asarray([(1, 2)])
    assert_raise_message(ValueError, "The number of clusters should not exceed the number of data points",
                         k_means.k_means, X, 2)


def two_points():
    X = np.asarray([(0, 0), (0, 1)])
    clusters, centers = k_means.k_means(X, 2)
    numpy.testing.assert_array_equal(clusters.tolist(), [0, 1])
    numpy.testing.assert_array_equal(centers, X)


def four_points():
    X = np.asarray([(0, 1), (0, 2), (0, 10), (0, 11)])
    clusters, centers = k_means.k_means(X, 2)
    numpy.testing.assert_array_equal(clusters.tolist(), [0, 0, 1, 1])


def separate_center():
    X = np.asarray([(0, 1), (0, 2), (0, 3), (0, 10)])
    clusters, centers = k_means.k_means(X, 2, np.asarray([(0, 2), (0, 10)]))
    numpy.testing.assert_array_equal(clusters.tolist(), [0, 0, 0, 1])
    numpy.testing.assert_array_equal(centers.tolist(), [[0, 2], [0, 10]])


def compare_to_sklearn():
    n_dim = 2
    n_clusters = 3
    n_samples = 5
    random_state = check_random_state(0)
    X = np.ndarray([n_clusters*n_samples, n_dim])
    initial_centers = random_state.rand(n_clusters, n_dim)*10
    for i in range(0, n_clusters):
        x_current = random_state.multivariate_normal(initial_centers[i, :], [[1, 0], [0, 1]], n_samples)
        X[i*n_samples:(i+1)*n_samples, :] = x_current
    closest_center, initial_centers = k_means.k_means(X, n_clusters)
    k_means_scikit_learn = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    closest_center_sklearn = k_means_scikit_learn.fit_predict(X)
    _swap_values_in_ndarray(closest_center_sklearn, 2, 0)
    numpy.testing.assert_array_equal(closest_center, closest_center_sklearn)


def _swap_values_in_ndarray(x, val1, val2):
    i_val1 = x == val1
    i_val2 = x == val2
    x[i_val1] = -1
    x[i_val2] = val1
    x[i_val1] = val2


if __name__ == '__main__':
    no_points()
    one_point()
    two_points()
    four_points()
    separate_center()
    compare_to_sklearn()
