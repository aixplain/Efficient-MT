import os
from typing import List, Sequence, Union

import numpy as np
import pandas as pd

myInput = Union[pd.DataFrame, np.ndarray, list]


def create_dirs_for_dataset(name):
    os.makedirs(f"./data/{name}", exist_ok=True)
    os.makedirs(f"./data/{name}/models", exist_ok=True)
    os.makedirs(f"./data/{name}/samples", exist_ok=True)
    os.makedirs(f"./data/{name}/segments", exist_ok=True)
    os.makedirs(f"./data/{name}/segments/embeddings", exist_ok=True)


def data_vstack(blocks: Sequence[myInput]) -> myInput:
    """
    Stack vertically sparse/dense arrays and pandas data frames.
    Args:
        blocks: Sequence of myInput objects.
    Returns:
        New sequence of vertically stacked elements.
    """
    if isinstance(blocks[0], pd.DataFrame):
        return blocks[0].append(blocks[1:])
    elif isinstance(blocks[0], np.ndarray):
        return np.concatenate(blocks)
    elif isinstance(blocks[0], list):
        return np.concatenate(blocks).tolist()

    raise TypeError("%s datatype is not supported" % type(blocks[0]))


def data_hstack(blocks: Sequence[myInput]) -> myInput:
    """
    Stack horizontally sparse/dense arrays and pandas data frames.
    Args:
        blocks: Sequence of myInput objects.
    Returns:
        New sequence of horizontally stacked elements.
    """
    if isinstance(blocks[0], pd.DataFrame):
        pd.concat(blocks, axis=1)
    elif isinstance(blocks[0], np.ndarray):
        return np.hstack(blocks)
    elif isinstance(blocks[0], list):
        return np.hstack(blocks).tolist()

    TypeError("%s datatype is not supported" % type(blocks[0]))


def add_row(X: myInput, row: myInput):
    """
    Returns X' =
    [X
    row]
    """
    if isinstance(X, np.ndarray):
        return np.vstack((X, row))
    elif isinstance(X, list):
        return np.vstack((X, row)).tolist()

    # data_vstack readily supports stacking of matrix as first argument
    # and row as second for the other data types
    return data_vstack([X, row])


def retrieve_rows(
    X: myInput, I: Union[int, List[int], np.ndarray]
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Returns the rows I from the data set X
    For a single index, the result is as follows:
    * 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame
    * row in case of list or numpy format
    """
    if isinstance(X, pd.DataFrame):
        return X.iloc[I]
    elif isinstance(X, np.ndarray):
        return X[I]
    elif isinstance(X, list):
        return np.array(X)[I].tolist()

    raise TypeError("%s datatype is not supported" % type(X))


def drop_rows(
    X: myInput, I: Union[int, List[int], np.ndarray]
) -> Union[np.ndarray, pd.DataFrame]:
    """
    Returns X without the row(s) at index/indices I
    """
    if isinstance(X, pd.DataFrame):
        return X.drop(I, axis=0)
    elif isinstance(X, np.ndarray):
        return np.delete(X, I, axis=0)
    elif isinstance(X, list):
        return np.delete(X, I, axis=0).tolist()

    raise TypeError("%s datatype is not supported" % type(X))


def enumerate_data(X: myInput):
    """
    for i, x in enumerate_data(X):
    Depending on the data type of X, returns:
    * A 1xM matrix in case of scipy sparse NxM matrix X
    * pandas series in case of a pandas data frame X
    * row in case of list or numpy format
    """
    if isinstance(X, pd.DataFrame):
        return X.iterrows()
    elif isinstance(X, np.ndarray) or isinstance(X, list):
        # numpy arrays and lists can readily be enumerated
        return enumerate(X)

    raise TypeError("%s datatype is not supported" % type(X))


def data_shape(X: myInput):
    """
    Returns the shape of the data set X
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
        # scipy.sparse, pandas and numpy all support .shape
        return X.shape
    elif isinstance(X, list):
        return np.array(X).shape

    raise TypeError("%s datatype is not supported" % type(X))
