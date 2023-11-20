from typing import Callable, Optional

import numpy as np
from flaml import AutoML
from sklearn.metrics import accuracy_score

from data_centric.uncertainty import uncertainty_sampling
from data_centric.utils.data import myInput

from .base import BaseLearner

"""
Classes for active learning algorithms
--------------------------------------
"""


class ActiveLearner(BaseLearner):
    """
    This class is an abstract model of a general active learning algorithm.
    Args:
        estimator: The estimator to be used in the active learning loop.
        embedding_pipeline: aiXplain Platform pileline url or name that we use for getting embeddings, If not specified, it will use X_training directly
        score_func: Function used for calculating performance metric,
        query_strategy: Function providing the query strategy for the active learning loop,
            for instance, active_learning.uncertainty.uncertainty_sampling.
        X_training: Initial training samples, if available.
        y_training: Initial training labels corresponding to initial training samples.
        bootstrap_init: If initial training data is available, bootstrapping can be done during the first training.
            Useful when building Committee models with bagging.
        on_transformed: Whether to transform samples with the pipeline defined by the estimator
            when applying the query strategy.
        **automl_settings: keyword arguments.
    Attributes:
        estimator: The estimator to be used in the active learning loop.
        query_strategy: Function providing the query strategy for the active learning loop.
        score_func:Function used for calculating performance metric.
        X_training: If the model hasn't been fitted yet it is None, otherwise it contains the samples
            which the model has been trained on. If provided, the method fit() of estimator is called during __init__()
        y_training: The labels corresponding to X_training.
    Examples:
        >>> from sklearn.datasets import load_iris
        >>> from flaml import AutoML
        >>> from data_centric.models import ActiveLearner
        >>> iris = load_iris()
        >>> # give initial training examples
        >>> X_training = iris['data'][[0, 50, 100]]
        >>> y_training = iris['target'][[0, 50, 100]]
        >>> automl_settings = {
        ...     "time_budget": 1,
        ...     "estimator_list": ['lgbm']
        ... }
        >>> # initialize active learner
        >>> learner = ActiveLearner(
        ...     estimator=AutoML(),
        ...     embedding_pipeline = "asdadas",
        ...     X_training=X_training, y_training=y_training, **automl_settings
        ... )
        >>>
        >>> # querying for labels
        >>> query_idx, query_sample = learner.query(iris['data'])
        >>>
        >>> # ...obtaining new labels from User...
        >>>
        >>> # teaching newly labelled examples
        >>> learner.teach(
        ...     X=iris['data'][query_idx].reshape(1, -1),
        ...     y=iris['target'][query_idx].reshape(1, ),
        ...     **automl_settings
        ... )
    """

    def __init__(
        self,
        estimator: AutoML,
        embedding_pipeline: str,
        query_strategy: Callable = uncertainty_sampling,
        score_func: Callable = accuracy_score,
        X_training: Optional[myInput] = None,
        y_training: Optional[myInput] = None,
        bootstrap_init: bool = False,
        on_transformed: bool = False,
        **automl_settings
    ) -> None:
        super().__init__(
            estimator,
            embedding_pipeline,
            query_strategy,
            score_func,
            X_training,
            y_training,
            bootstrap_init,
            on_transformed,
            **automl_settings
        )

    def teach(
        self,
        X: myInput,
        y: myInput,
        bootstrap: bool = False,
        only_new: bool = False,
        **automl_settings
    ) -> None:
        """
        Adds X and y to the known training data and retrains the predictor with the augmented dataset.
        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, training is done on a bootstrapped dataset. Useful for building Committee models
                with bagging.
            only_new: If True, the model is retrained using only X and y, ignoring the previously provided examples.
                Useful when working with models where the .fit() method doesn't retrain the model from scratch (e. g. in
                tensorflow or keras).
            **fit_kwargs: Keyword arguments to be passed to the fit method of the predictor.
        """
        self._add_training_data(X, y)
        if not only_new:
            self._fit_to_known(bootstrap=bootstrap, **automl_settings)
        else:
            self._fit_on_new(X, y, bootstrap=bootstrap, **automl_settings)
