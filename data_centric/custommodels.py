from tkinter import X
from vowpalwabbit.pyvw import vw
import numpy as np
from typing import Any, Callable, Optional, Tuple, Union

from data_centric.utils.data import myInput, retrieve_rows
from sklearn.utils import check_X_y
import warnings


class VWModel:
    def __init__(
        self,
        estimator_parameter_string: str,
        training_feature_formating_fn=None,
        test_feature_foramtting_fn=None,
        query_strategy: Callable = None,
        X_init: Optional[myInput] = None,
        y_init: Optional[myInput] = None,
        **kwargs
    ):

        self.estimator = vw(estimator_parameter_string, **kwargs)
        self.training_feature_formating_fn = training_feature_formating_fn
        self.test_feature_foramtting_fn = test_feature_foramtting_fn
        self.query_strategy = query_strategy
        self.X_init = X_init
        self.y_init = y_init

        if X_init is not None and y_init is not None:
            self.learn(X_init, y_init)

    def learn(self, features, y):
        check_X_y(
            features,
            y,
            accept_sparse=False,
            ensure_2d=False,
            allow_nd=True,
            multi_output=True,
            dtype=None,
        )
        for xi, yi in zip(features, y):
            self.estimator.learn(self.training_feature_formating_fn(xi, yi))

    def predict(self, features):
        probs = self.predict_proba(features)
        return np.argmax(probs, axis=1)

    def predict_proba(self, features):
        probs = []
        for _ in features:
            probs.append(self.estimator.predict(self.test_feature_foramtting_fn(_)))
        return np.array(probs)

    def query(self, X_pool, *query_args, **query_kwargs) -> Union[Tuple, myInput]:
        """
        Finds the n_instances most informative point in the data provided by calling the query_strategy function.
        Args:
            X_pool: Pool of unlabeled instances to retrieve most informative instances from
            *query_args: The arguments for the query strategy. For instance, in the case of
                :func:`~modAL.uncertainty.uncertainty_sampling`, it is the pool of samples from which the query strategy
                should choose instances to request labels.
            **query_kwargs: Keyword arguments for the query strategy function.
        Returns:
            Value of the query_strategy function. Should be the indices of the instances from the pool chosen to be
            labelled and the instances themselves. Can be different in other cases, for instance only the instance to be
            labelled upon query synthesis.
        """
        query_result = self.query_strategy(self, X_pool, *query_args, **query_kwargs)

        if isinstance(query_result, tuple):
            warnings.warn(
                "Query strategies should no longer return the selected instances, "
                "this is now handled by the query method. "
                "Please return only the indices of the selected instances.",
                DeprecationWarning,
            )
            return query_result

        return query_result, retrieve_rows(X_pool, query_result)
