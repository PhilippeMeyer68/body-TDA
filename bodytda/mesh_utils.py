import meshio


def scan_from_index(path_scan):
    """
    Load a 3D scan from a file and return its points as a list.

    Parameters
    ----------
    path_scan : str
        The path to the 3D scan file.

    Returns
    -------
    list of list of float
        A list of points from the 3D scan, where each point is represented as a list of coordinates.
    """

    scan = meshio.read(path_scan)
    scan = scan.points
    scan = scan.tolist()
    return scan


def scan_height(scan):
    """
    Calculate the height of a 3D scan, defined as the difference between the maximum and minimum z-coordinates.

    Parameters
    ----------
    scan : list of list of float
        A list of points from the 3D scan, where each point is represented as a list of coordinates.

    Returns
    -------
    float
        The height of the 3D scan.
    """

    z = [row[2] for row in scan]
    return max(z) - min(z)


def scan_normalization(scan, target_height=1.7):
    """
    Normalize the height of a 3D scan to a target value.

    Parameters
    ----------
    scan : list of list of float
        A list of points from the 3D scan, where each point is represented as a list of coordinates.

    target_height : float, optional (default: 1.7)
        The target height to which the scan will be normalized.

    Returns
    -------
    list of list of float
        A normalized 3D scan where the z-coordinates are scaled to the target height.
    """

    scantemp = scan
    coeff = target_height / scan_height(scan)
    for i in range(len(scan)):
        for j in range(3):
            scantemp[i][j] = scantemp[i][j] * coeff
    return scantemp
