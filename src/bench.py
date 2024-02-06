"""
Numpy-style Documentation for the `bench` Module

This module provides functionality to benchmark the training and inference times
of various scikit-learn machine learning models.

Functions
---------
train_model(reg, X_train, y_train, T, score)
    Train and evaluate the performance of a machine learning model.

bench(num_samples, num_features, fix_comp_time, reg_or_cls="reg", nb_output=1)
    Benchmark the training and inference times of scikit-learn models.

"""

from typing import *
import time
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin
import numpy as np

DISPLAY_WARNING = False
THRESHOLD_DECIMAL = 10
ROUNDING = 3
CENTER_TABLE_DISPLAY = True


def _bench_func(func: Callable, T: int, *func_args):
    time_count = 0
    start_time = time.time()
    while start_time + T > time.time():
        func(*func_args)
        time_count += 1
    end_time = time.time()

    if time_count <= THRESHOLD_DECIMAL:
        data_samples_ingested_per_second = (end_time - start_time) / time_count
        time_count = round(data_samples_ingested_per_second / T, ROUNDING)

    return time_count


def _bench_model(model_constructor: callable, X: np.ndarray, y: np.ndarray, T: int, score: List[Tuple]) -> None:
    """
    Train and evaluate the performance of a machine learning model.

    Parameters
    ----------
    model_constructor : callable
        A constructor for the machine learning model.
    X : numpy.ndarray
        The input training data.
    y : numpy.ndarray
        The target training data.
    T : float
        The maximum allowed time for training in seconds.
    score : list
        A list to store the performance metrics of each model.

    Returns
    -------
    None
    """

    try:
        model = model_constructor()
        model_name = model.__class__.__name__
    except Exception as e:
        if DISPLAY_WARNING:
            print(f"ERROR with {model_constructor.__name__} at constructor time: {e}")
        return

    try:
        training_time = _bench_func(model.fit, T, X, y)
    except Exception as e:
        if DISPLAY_WARNING:
            print(f"ERROR with {model_name} at training time: {e}")
        return

    try:
        inference_time = _bench_func(model.predict, T, X)
    except Exception as e:
        if DISPLAY_WARNING:
            print(f"ERROR with {model_name} at inference time: {e}")
        inference_time = "N/A"

    score.append((model_name, training_time, inference_time))


def _display(score:List[Tuple])->None:
    if len(score) < 1:
        print("No score table to display")
        return

    if CENTER_TABLE_DISPLAY:
        # compute the number of chars per column
        number_of_chars_per_column = [0] * len(score[0])
        for i, row in enumerate(score):
            for j, val in enumerate(row):
                nb_char_val = len(str(val))
                number_of_chars_per_column[j] = max(nb_char_val, number_of_chars_per_column[j])

        # Compute each row
        white_space_sep = 1
        rows = []
        for i, row in enumerate(score):
            row_text = ""
            for j, val in enumerate(row):
                val_str = str(val)
                nb_white_space = ((number_of_chars_per_column[j] - len(val_str)) + white_space_sep)
                row_text += val_str
                row_text += " " * nb_white_space
            rows.append(row_text)

        # Print the rows
        for r in rows:
            print(r)
    else:
        for s in score:
            print(' '.join(map(str, s)))


def bench(num_samples: int, num_features: int, fix_comp_time: float, reg_or_cls: str = "cls",
          nb_cls_output: int = 2, nb_reg_output: int = 1, sorting_mode=1) -> List[Tuple]:
    """
    Benchmark the training and inference times of scikit-learn models.

    Parameters
    ----------
    num_samples : int
        The number of training samples.
    num_features : int
        The number of features in the training data.
    fix_comp_time : float
        The minimum allowed time for training in seconds.
    reg_or_cls : str
        Either 'reg' for regression models or 'cls' for classification models.
    nb_cls_output : int, optional
        The number of output classes when reg_or_cls=='cls', by default 2.
    nb_reg_output : int, optional
        The number of output when reg_or_cls=='reg', by default 1.
    sorting_mode : int, optional
        0 sorting according to algorithm name, 1 according to training time, 2 inference time

    Returns
    -------
    None
    """
    if not DISPLAY_WARNING:
        import warnings
        for warn in [DeprecationWarning, FutureWarning, UserWarning, RuntimeWarning, ConvergenceWarning]:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=ConvergenceWarning)

    score = []

    assert (reg_or_cls in {"reg", "cls"})
    mixin = RegressorMixin if reg_or_cls == "reg" else ClassifierMixin
    model_constructors = [est for est in all_estimators() if issubclass(est[1], mixin)]

    X_train = np.zeros((num_samples, num_features))

    if mixin == RegressorMixin:
        y_train = np.ones((len(X_train), nb_reg_output)) * 0.5
    else:
        y_train = np.random.randint(low=0, high=nb_cls_output, size=(len(X_train),))  # sparse representation

    for num_samples, model_constructor in model_constructors:
        _bench_model(model_constructor, X_train, y_train, fix_comp_time, score)

    score.sort(key=lambda x: x[sorting_mode])

    _display(score)
    return score


if __name__ == "__main__":
    bench(num_samples=100, num_features=10, fix_comp_time=1, reg_or_cls="cls")
