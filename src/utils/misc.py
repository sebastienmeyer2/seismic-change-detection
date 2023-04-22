"""Miscellaneous utils."""


import os

import argparse


def create_if_no_dir(
    dir_name: str
):
    """Create a directory if it does not exist.

    Parameters
    ----------
    dir_name : str
        Name of the directory.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def raise_if_no_dir(
    dir_name: str
):
    """Raise error if directory does not exist.

    Parameters
    ----------
    dir_name : str
        Name of the directory.

    Raises
    ------
    OSError
        If the directory does not exist.
    """
    if not os.path.exists(dir_name):

        err_msg = f"The data directory {dir_name} does not exist on disk."
        raise OSError(err_msg)


def str2bool(
    v: str
) -> bool:
    """An easy way to handle boolean options.

    Parameters
    ----------
    v : str
        Argument value.

    Returns
    -------
    str2bool(v) : bool
        Corresponding boolean value, if it exists.

    Raises
    ------
    argparse.ArgumentTypeError
        If the entry cannot be converted to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def float_zero_one(
    v: float
) -> float:
    """Check that the value is between zero and one.

    Parameters
    ----------
    v : float
        Value to test.

    Returns
    -------
    v : float
        Value if between zero and one.

    Raises
    ------
    argparse.ArgumentTypeError
        If the entry is not between zero and one.
    """
    if 1. >= v >= 0.:
        return v
    raise argparse.ArgumentTypeError("Argument must be between 0 and 1.")
