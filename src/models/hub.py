"""Wrapper around *ruptures* package."""


from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ruptures import KernelCPD
from ruptures.metrics import hausdorff, precision_recall, randindex


def fit_predict(
    signal: Optional[np.ndarray] = None,
    cost_name: str = "rbf",
    cost_params: Optional[Dict[str, Any]] = None,
    min_size: int = 2,
    n_bkps: Optional[int] = None,
    pen: Optional[float] = None
) -> List[int]:
    """Fit and predict a model from *ruptures* package.

    Parameters
    ----------
    signal : optional np.ndarray, default=None
        Univariate or multivariate signal for change detection.

    cost_name : str, default="rbf"
        Name of the cost function to use following *ruptures* package usage.

    cost_params : optional dict of {str: any}, default=None
        Dictionary of parameters for the cost function.

    min_size : int, default=2
        Minimum segment length.

    n_bkps : optional int, default=None
        If provided, the `Dynp` algorithm will be used to predict exactly **n_bkps** breakpoints.
        Otherwise, the `Pelt` algorithm will be used with the given **pen** penalty.

    pen : optional float, default=None
        If **n_bkps** is None, then penalty value for the `Pelt` algorithm.

    Returns
    -------
    pred_bkps : list of int
        List of predicted indices for the breakpoints in the signal.
    """
    if signal is None:
        return []

    algo = KernelCPD(kernel=cost_name, min_size=min_size, params=cost_params)

    algo = algo.fit(signal)

    pred_bkps = algo.predict(n_bkps=n_bkps, pen=pen)

    return pred_bkps


def evaluate(
    true_bkps: List[int],
    pred_bkps: List[int],
    margin: int = 10
) -> Dict[str, float]:
    """Evaluate predicted breakpoints.

    Parameters
    ----------
    true_bkps : list of int
        List of true indices for the breakpoints in the signal.

    pred_bkps : list of int
        List of predicted indices for the breakpoints in the signal.

    Returns
    -------
    metrics_scores : dict of {str: float}
        Dictionary containing the corresponding values of *ruptures* evaluation metrics.
    """
    metrics_scores = {
        "precision-recall": precision_recall(true_bkps, pred_bkps, margin=margin),
        "hausdorff": hausdorff(true_bkps, pred_bkps),
        "randindex": randindex(true_bkps, pred_bkps)
    }

    return metrics_scores


def elbow_method(
    signal: Optional[np.ndarray] = None,
    cost_name: str = "rbf",
    cost_params: Optional[Dict[str, Any]] = None,
    min_size: int = 2,
    n_bkps_max: int = 20
) -> Tuple[np.ndarray, ...]:
    """Run elbow method to find the best number of breakpoints.

    Parameters
    ----------
    signal : optional np.ndarray, default=None
        Univariate or multivariate signal for change detection.

    cost_name : str, default="rbf"
        Name of the cost function to use following *ruptures* package usage.

    cost_params : optional dict of {str: any}, default=None
        Dictionary of parameters for the cost function.

    min_size : int, default=2
        Minimum segment length.

    n_bkps_max : int, default=20
        Maximal number of breakpoints to compute the elbow method.

    Returns
    -------
    array_of_n_bkps : np.ndarray
        List of number of breakpoints from 1 to **n_bkps_max**.

    array_of_costs : np.ndarray
        List of corresponding sum of costs when predicting using `Dynp` algorithm.
    """
    array_of_n_bkps = np.arange(1, n_bkps_max + 1)

    array_of_costs = np.zeros(n_bkps_max)

    if signal is None:
        return array_of_n_bkps, array_of_costs

    # Fit the algorithm
    algo = KernelCPD(kernel=cost_name, min_size=min_size, params=cost_params)

    algo = algo.fit(signal)

    # From dynamic programming, we smaller number of breakpoints for free
    _ = algo.predict(n_bkps=n_bkps_max)

    for i, n_bkps in enumerate(array_of_n_bkps):

        pred_bkps = algo.predict(n_bkps=n_bkps)

        array_of_costs[i] = algo.cost.sum_of_costs(pred_bkps)

    return array_of_n_bkps, array_of_costs


def dp_bkps(
    signal: Optional[np.ndarray] = None,
    cost_name: str = "rbf",
    cost_params: Optional[Dict[str, Any]] = None,
    min_size: int = 2,
    n_bkps_max: int = 20
) -> Tuple[np.ndarray, List[List[int]]]:
    """Return all sets of breakpoints until a maximal number using Dynp.

    Parameters
    ----------
    signal : optional np.ndarray, default=None
        Univariate or multivariate signal for change detection.

    cost_name : str, default="rbf"
        Name of the cost function to use following *ruptures* package usage.

    cost_params : optional dict of {str: any}, default=None
        Dictionary of parameters for the cost function.

    min_size : int, default=2
        Minimum segment length.

    n_bkps_max : int, default=20
        Maximal number of breakpoints to compute the elbow method.

    Returns
    -------
    array_of_n_bkps : np.ndarray
        List of number of breakpoints from 1 to **n_bkps_max**.

    list_of_bkps : list of list of int
        List of corresponding breakpoints indices.
    """
    array_of_n_bkps = np.arange(1, n_bkps_max + 1)

    list_of_bkps = []

    if signal is None:
        return array_of_n_bkps, list_of_bkps

    # Fit the algorithm
    algo = KernelCPD(kernel=cost_name, min_size=min_size, params=cost_params)

    algo = algo.fit(signal)

    # From dynamic programming, we get smaller number of breakpoints for free
    _ = algo.predict(n_bkps=n_bkps_max)

    for n_bkps in array_of_n_bkps:

        pred_bkps = algo.predict(n_bkps=n_bkps)

        list_of_bkps.append(pred_bkps)

    return array_of_n_bkps, list_of_bkps


def p_bkps(
    signal: Optional[np.ndarray] = None,
    cost_name: str = "rbf",
    cost_params: Optional[Dict[str, Any]] = None,
    min_size: int = 2,
    n_pts: int = 20,
    min_pen: Optional[float] = None,
    max_pen: float = 1.0
) -> Tuple[np.ndarray, List[List[int]]]:
    """Return all sets of breakpoints until a maximal penalty using Pelt.

    Parameters
    ----------
    signal : optional np.ndarray, default=None
        Univariate or multivariate signal for change detection.

    cost_name : str, default="rbf"
        Name of the cost function to use following *ruptures* package usage.

    cost_params : optional dict of {str: any}, default=None
        Dictionary of parameters for the cost function.

    min_size : int, default=2
        Minimum segment length.

    n_pts : int, default=20
        Number of data points.

    min_pen : optional float, default=None
        Minimum penality to test.

    max_pen : float, default=1.0
        Maximum penality to test.

    Returns
    -------
    array_of_pens : np.ndarray
        List of penalities from **min_pen** to **max_pen**.

    list_of_bkps : list of list of int
        List of corresponding breakpoints indices.
    """
    if min_pen is None:
        min_pen = max_pen / n_pts

    array_of_pens = np.linspace(min_pen, max_pen, n_pts)

    list_of_bkps = []

    if signal is None:
        return array_of_pens, list_of_bkps

    # Fit the algorithm
    algo = KernelCPD(kernel=cost_name, min_size=min_size, params=cost_params)

    algo = algo.fit(signal)

    # With Pelt, specify the penalty gives the number of breakpoints
    _ = algo.predict(pen=max_pen)

    for pen in array_of_pens:

        pred_bkps = algo.predict(pen=pen)

        list_of_bkps.append(pred_bkps)

    return array_of_pens, list_of_bkps
