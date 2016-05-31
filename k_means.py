import numpy as np
from sklearn.utils import check_random_state


def k_means(x, k, centers=None, random_state=0):
    """Distribute the data from x into k clusters
    :param x: np.ndarray containing the datapoints
    :param k: int is the number of clusters
    :param centers: np.ndarray allows to specify the initial positions of centers
    :random_state: int is the seed for the random number generator"""

    if x.ndim != 2:
        n_features = 1
        for i in range(1, x.ndim):
            n_features *= x.shape[i]
        x = x.copy().reshape(x.shape[0], n_features)

    for i in range(x.ndim):
        if x.shape[i] == 0:
            raise ValueError("The input array should not contain any singleton dimensions")

    if k > x.shape[0]:
        raise ValueError("The number of clusters should not exceed the number of data points")

    # We want the random state to be repeatable (for the unit tests)
    random_state = check_random_state(random_state)
    if centers is None:
        centers = x[random_state.randint(0, high=x.shape[0], size=k)]

    sum_of_distances = np.inf
    while True:
        previous_sum_of_distances = sum_of_distances
        closest_center, sum_of_distances = _get_nearest_center(x, centers)
        for i_center in range(centers.shape[0]):
            centers[i_center, :] = np.mean(x[closest_center == i_center, :], axis=0)
        if sum_of_distances >= previous_sum_of_distances:
            return closest_center, centers


def _get_nearest_center(x, centers):
    """For each point, find the closest center"""

    distance_to_center = np.zeros([x.shape[0], centers.shape[0]])
    for i_center in range(centers.shape[0]):
        distance_to_center[:, i_center] = np.linalg.norm(x - centers[i_center], axis=1)
    closest_center = np.argmin(distance_to_center, axis=1)
    j = np.sum(np.min(distance_to_center, axis=1))
    return closest_center, j
