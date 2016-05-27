import numpy as np
import matplotlib.pyplot as plt
import k_means
import cProfile, pstats, io

perform_profiling = 0  # SWITCH BETWEEN PROFILING WITH MANY DATAPOINTS, AND PLOTTING WITH FEW OF THEM
n_dim = 2
if perform_profiling:
    n_clusters = 10
    n_samples = 10000
else:
    n_clusters = 3
    n_samples = 5

X = np.ndarray([n_clusters*n_samples, n_dim])
print("Plotting the initial distribution of points")
colors = np.linspace(0, 1, n_clusters)
markers = ['o', '*', 'H', 's', 'v', 'p', '>', '.', 'D', 'h', 'x', 'd', ',', '_', '^']
initial_centers = np.random.rand(n_clusters, n_dim)*10
for clr, i in zip(colors, range(0, n_clusters)):
    x_current = np.random.multivariate_normal(initial_centers[i, :], [[1, 0], [0, 1]], n_samples)
    X[i*n_samples:(i+1)*n_samples, :] = x_current
    if not perform_profiling:
        plt.scatter(x_current[:, 0], x_current[:, 1], s=1000, c=plt.cm.viridis(clr), marker=markers[i % len(markers)])

if not perform_profiling:
    print("Points:")
    print(X)

if perform_profiling:
    pr = cProfile.Profile()
    pr.enable()

print('\nCalculating clusters\n')
asked_for_n_clusters = 3
colors = np.linspace(0, 1, asked_for_n_clusters)
closest_center, initial_centers = k_means.k_means(X, asked_for_n_clusters)

if perform_profiling:
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
else:
    print("Closest centers:")
    print(closest_center)

    print("Initial centers:")
    print(initial_centers)

    print("Plotting the calculated clusters")
    for clr, i in zip(colors, range(0, asked_for_n_clusters)):
        plt.scatter(X[closest_center == i, 0], X[closest_center == i, 1],
                        c='k', s=5000, alpha=0.3, marker=markers[i % len(markers)])
    plt.show()
