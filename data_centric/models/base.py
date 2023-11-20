"""
Base classes for active learning algorithms
------------------------------------------
"""
import abc
import sys
import warnings
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from flaml import AutoML
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.utils import check_X_y

from data_centric.utils.data import data_hstack, data_vstack, myInput, retrieve_rows


ABC = abc.ABC


class BaseLearner:
    def __init__(
        self,
        estimator: AutoML,
        embedding_pipeline: str,
        query_strategy: Callable,
        score_func: Callable,
        X_training: Optional[myInput] = None,
        y_training: Optional[myInput] = None,
        bootstrap_init: bool = False,
        on_transformed: bool = False,
        **automl_settings
    ) -> None:
        self.estimator = estimator
        self.embedding_pipeline = embedding_pipeline
        self.query_strategy = query_strategy
        self.score_func = score_func
        self.on_transformed = (
            on_transformed  ## TODO: Will be used to get embeddings later
        )

        self.iteration_id = 0
        self.stats = {}  # type: ignore
        self.X_training = X_training
        self.y_training = y_training
        if X_training is not None:
            self._fit_to_known(bootstrap=bootstrap_init, **automl_settings)

    def _add_training_data(self, X: myInput, y: myInput) -> None:
        """
        Adds the new data and label to the known data, but does not retrain the model.
        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
        Note:
            If the estimator has been fitted, the features in X have to agree with the training samples which the
            estimator has seen.
        """
        check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=False,
            allow_nd=True,
            multi_output=True,
            dtype=None,
        )

        if self.X_training is None:
            self.X_training = X
            self.y_training = y
        else:
            try:
                self.X_training = data_vstack((self.X_training, X))
                self.y_training = data_vstack((self.y_training, y))
            except ValueError:
                raise ValueError(
                    "the dimensions of the new training data and label must"
                    "agree with the training data and labels provided so far"
                )

    def run_pipeline_on_input(self, X: myInput) -> np.ndarray:
        """
        Transforms the data as supplied to the estimator.
        * In case the estimator is an skearn pipeline, it applies all pipeline components but the last one.
        * In case the estimator is an ensemble, it concatenates the transformations for each classfier
            (pipeline) in the ensemble.
        * Otherwise returns the non-transformed dataset X
        Args:
            X: dataset to be transformed
        Returns:
            Transformed data set
        """
        Xt = []
        pipes = (
            self.embedding_pipeline
        )  ## TODO: get pipeline executor here # type: ignore

        ################################
        # transform data with pipelines used by embedder
        for pipe in pipes:
            ## TODO: run pipelines here
            Xt.append(pipe.transform(X))  # type: ignore

        # in case no transformation pipelines are used by the estimator,
        # return the original, non-transfored data
        if not Xt:
            return X

        ################################
        # concatenate all transformations and return
        return data_hstack(Xt)

    def _fit_to_known(
        self, bootstrap: bool = False, **automl_settings
    ) -> "BaseLearner":
        """
        Fits self.estimator to the training data and labels provided to it so far.
        Args:
            bootstrap: If True, the method trains the model on a set bootstrapped from the known training instances.
            **automl_settings: Keyword arguments to be passed to the fit method of the predictor.
        Returns:
            self
        """
        if not bootstrap:
            print(type(self.X_training))
            self.estimator.fit(self.X_training, self.y_training, **automl_settings)
        else:
            n_instances = self.X_training.shape[0]
            bootstrap_idx = np.random.choice(
                range(n_instances), n_instances, replace=True
            )
            self.estimator.fit(
                self.X_training[bootstrap_idx],
                self.y_training[bootstrap_idx],
                **automl_settings
            )

        return self

    def _fit_on_new(
        self, X: myInput, y: myInput, bootstrap: bool = False, **automl_settings
    ) -> "BaseLearner":
        """
        Fits self.estimator to the given data and labels.
        Args:
            X: The new samples for which the labels are supplied by the expert.
            y: Labels corresponding to the new instances in X.
            bootstrap: If True, the method trains the model on a set bootstrapped from X.
            **automl_settings: Keyword arguments to be passed to the fit method of the predictor.
        Returns:
            self
        """
        check_X_y(
            X,
            y,
            accept_sparse=True,
            ensure_2d=False,
            allow_nd=True,
            multi_output=True,
            dtype=None,
            force_all_finite=self.force_all_finite,
        )

        if not bootstrap:
            self.estimator.fit(X, y, **automl_settings)
        else:
            bootstrap_idx = np.random.choice(
                range(X.shape[0]), X.shape[0], replace=True
            )
            self.estimator.fit(X[bootstrap_idx], y[bootstrap_idx])

        return self

    def fit(
        self, X: myInput, y: myInput, bootstrap: bool = False, **automl_settings
    ) -> "BaseLearner":
        """
        Interface for the fit method of the predictor. Fits the predictor to the supplied data, then stores it
        internally for the active learning loop.
        Args:
            X: The samples to be fitted.
            y: The corresponding labels.
            bootstrap: If true, trains the estimator on a set bootstrapped from X.
                Useful for building Committee models with bagging.
            **automl_settings: Keyword arguments to be passed to the fit method of the predictor.
        Note:
            When using scikit-learn estimators, calling this method will make the ActiveLearner forget all training data
            it has seen!
        Returns:
            self
        """
        check_X_y(
            X,
            y,
            accept_sparse=True,
            ensure_2d=False,
            allow_nd=True,
            multi_output=True,
            dtype=None,
            force_all_finite=self.force_all_finite,
        )
        self.X_training, self.y_training = X, y
        return self._fit_to_known(bootstrap=bootstrap, **automl_settings)

    def predict(self, X: myInput, **predict_kwargs) -> Any:
        """
        Estimator predictions for X. Interface with the predict method of the estimator.
        Args:
            X: The samples to be predicted.
            **predict_kwargs: Keyword arguments to be passed to the predict method of the estimator.
        Returns:
            Estimator predictions for X.
        """
        return self.estimator.predict(X, **predict_kwargs)

    def predict_proba(self, X: myInput, **predict_proba_kwargs) -> Any:
        """
        Class probabilities if the predictor is a estimator. Interface with the predict_proba method of the estimator.
        Args:
            X: The samples for which the class probabilities are to be predicted.
            **predict_proba_kwargs: Keyword arguments to be passed to the predict_proba method of the estimator.
        Returns:
            Class probabilities for X.
        """
        return self.estimator.predict_proba(X, **predict_proba_kwargs)

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

    def update_query_strategy(self, query_strategy: Callable) -> None:
        """
        Updates the query strategy function.
        Args:
            query_strategy: The new query strategy function.
        """
        self.query_strategy = query_strategy

    def score(self) -> Any:
        """
        Interface for the score method of the predictor.
        Returns:
            The lowest loss of the estimator on the training set
        """
        y_pred = self.predict(self.X_training)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_training, y_pred, average="weighted"
        )

        self.stats = {
            "y_true": list([str(_) for _ in self.y_training]),
            "y_pred": list([str(_) for _ in y_pred]),
            "accuracy": float(
                round(balanced_accuracy_score(self.y_training, y_pred), 2)
            ),
            "precision": float(round(precision, 2)),
            "recall": float(round(recall, 2)),
            "f1": float(round(f1, 2)),
            "best_loss": float(round(self.estimator.best_loss, 2)),
        }
        return self.stats

    @abc.abstractmethod
    def teach(self, *args, **kwargs) -> None:
        pass
