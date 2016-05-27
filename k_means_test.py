import k_means
import numpy as np
import numpy.testing
from sklearn.utils.testing import assert_raise_message


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


if __name__ == '__main__':
    no_points()
    one_point()
    two_points()
    four_points()
    separate_center()