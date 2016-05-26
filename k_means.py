import numpy as np
from sklearn.utils import check_random_state


def k_means(x, k):
    """Distribute the data from x into k clusters"""

    for i in range(x.ndim):
        if x.shape[i] == 0:
            raise ValueError("The input array should not contain any singleton dimensions")

    if k > x.shape[0]:
        raise ValueError("The number of clusters should not exceed the number of data points")

    # We want the random state to be repeatable (for the unit tests)
    random_state = check_random_state(0)
    initial_center_indices = random_state.randint(0, high=x.shape[0], size=k)

    centers = x[initial_center_indices, :]
    j = np.inf
    while True:
        j_previous = j
        closest_center, j = _get_nearest_center(x, centers)
        for i_center in range(0, centers.shape[0]):
            if any(closest_center == i_center):
                centers[i_center, :] = np.mean(x[closest_center == i_center, :], axis=0)
            else:

                # For this center no point is the closest to this one. Let's regenerate a new center
                while True:
                    new_center_index = random_state.randint(0, high=x.shape[0], size=1)
                    if new_center_index not in initial_center_indices:
                        break
                    if len(initial_center_indices) > x.shape[0]:
                        raise RuntimeError("Cannot generate a new guess for a center")
                centers[i_center, :] = x[new_center_index, :]
                initial_center_indices.append(new_center_index)
        if j < j_previous:
            return closest_center, centers


def _get_nearest_center(x, centers):
    """For each point, find the closest center"""

    distance_to_center = np.zeros([x.shape[0], centers.shape[0]])
    for i_center in range(0, centers.shape[0]):
        distance_to_center[:, i_center] = np.linalg.norm(x - centers[i_center, :], axis=1)
    closest_center = np.argmin(distance_to_center, axis=1)

    # Calculate the overall scalar norm
    j = 0
    for i_center in range(0, x.shape[0]):

        # Don't recalculate the norm, reuse the one already calculated
        j += distance_to_center[i_center, closest_center[i_center]]
    return closest_center, j
