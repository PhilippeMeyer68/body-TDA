import gudhi as gd
import gudhi.wasserstein as gw
import matplotlib.pyplot as plt
import numpy as np


def persistence_diagram(point_cloud, dimension=2, min_persistence=0.0003):
    """
    Compute the persistence diagram of a point cloud using the Alpha complex.

    Parameters
    ----------
    point_cloud : list of list (shape: N × d)
        The input point cloud, where N is the number of points and d is the dimensionality.

    dimension : int, optional (default: 2)
        The maximum homology dimension to compute persistence intervals for.

    min_persistence : float, optional (default: 0.0003)
        The minimum persistence value to filter out short-lived topological features.

    Returns
    -------
    pdiagram : list of tuples
        The persistence diagram, containing pairs (birth, death) of topological features.

    pdiagram_decolor : list of numpy.ndarray
        A list where the i-th entry contains the persistence intervals for homology dimension i.
    """

    rips_complex = gd.AlphaComplex(points=point_cloud)
    simplex_tree = rips_complex.create_simplex_tree()
    pdiagram = simplex_tree.persistence(min_persistence=min_persistence)
    pdiagram_decolor = []
    for i in range(dimension + 1):
        pdiagram_decolor.append(simplex_tree.persistence_intervals_in_dimension(i))
    return pdiagram, pdiagram_decolor


def plot_persistence_diagram(pdiagram):
    """
    Plot the persistence diagram and barcode of a given persistence diagram.

    Parameters
    ----------
    pdiagram : list of tuples
        The persistence diagram, containing pairs (birth, death) of topological features.

    Returns
    -------
    None
        Displays the persistence diagram and barcode plot.
    """

    plt.figure()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    gd.plot_persistence_diagram(persistence=pdiagram, legend=True, axes=axes[0])
    gd.plot_persistence_barcode(pdiagram, axes=axes[1])
    plt.show()


def bottleneck_distance(pdiagram_decolor_1, pdiagram_decolor_2, dimension=2):
    """
    Compute the bottleneck distance between two persistence diagrams.

    Parameters
    ----------
    pdiagram_decolor_1 : list of tuples
        The first persistence diagram, where each entry corresponds to a specific homology dimension.

    pdiagram_decolor_2 : list of tuples
        The second persistence diagram, structured similarly to `pdiagram_decolor_1`.

    dimension : int, optional (default: 2)
        The maximum homology dimension to compute the bottleneck distance for.

    Returns
    -------
    float
        The maximum bottleneck distance across all computed homology dimensions.
    """

    distances = []
    for i in range(dimension + 1):
        distance = gd.bottleneck_distance(pdiagram_decolor_1[i], pdiagram_decolor_2[i])
        distances.append(distance)
    return max(distances)


def wasserstein_distance(pdiagram_decolor_1, pdiagram_decolor_2, p=2, order=2):
    """
    Compute the Wasserstein distance between two persistence diagrams.

    Parameters
    ----------
    pdiagram_decolor_1 : list of tuples
        The first persistence diagram, where each entry corresponds to a specific homology dimension.

    pdiagram_decolor_2 : list of tuples
        The second persistence diagram, structured similarly to `pdiagram_decolor_1`.

    p : float, optional (default: 2)
        The power used for the distance metric. Typically, p = 1 for the 1-Wasserstein distance and p = 2 for the 2-Wasserstein distance.

    order : int, optional (default: 2)
        The order of the Wasserstein distance, which determines the weight given to different features in the distance calculation.

    Returns
    -------
    float
        The Wasserstein distance between the two persistence diagrams.
    """

    distances = []
    for i in range(len(pdiagram_decolor_1)):
        distance = gw.wasserstein_distance(
            np.array(pdiagram_decolor_1[i]),
            np.array(pdiagram_decolor_2[i]),
            order=order,
            internal_p=p,
        )
        distances.append(distance)
    return np.power(sum(np.power(distance, p) for distance in distances), 1 / p)
