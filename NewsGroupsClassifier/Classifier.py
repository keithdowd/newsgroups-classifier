# -*- coding: utf-8 -*-

import numpy

from sklearn.model_selection import GridSearchCV

import time


class Classifier:
    """Implements classification.

    Base class that implements fit procedure for fitting a 
    classification or regression estimator using grid search. 
    Expects a pipeline object and parameters dict. 

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline object that 
            describes (1) feature extraction and (2) classification.
        parameters (dict): Dictionary that defines the parameter(s) and 
            values to search across for the Pipeline objects.

    """

    def __init__(self, pipeline, parameters, *args, **kwargs):

        try:
            self._classifier = GridSearchCV(
                pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
        except Exception as error:
            raise Exception(error)

    def __str__(self, *args, **kwargs):
        """Human friendly readable label for class instance.

        Prints name classifier signature stored in _classifier if 
        available and prints "No classifier name is set" otherwise.

        Returns:
            str: Name of classifier.

        """

        if self._classifier == "" or self._classifier is None:
            return "No classifier is set."

        return self._classifier

    def fit(self, examples, labels, *args, **kwargs):
        """Run fit with all sets of parameters.

        Run fit with all sets of parameters. Mask public method for __fit() private method.

        Args:
            examples (array-like [list, tuple]): Vector of [n_samples, n_features] where n_samples is the number of samples and n_features is the number of features.
            labels (array-like [list, tuple, numpy.ndarray]): Vector of targets relative to examples for classification or regression.

        """

        self.__fit(examples, labels)

    def __fit(self, examples, labels, *args, **kwargs):
        """Run fit with all sets of parameters.

        Run fit with all sets of parameters.

        Args:
            examples (array-like [list, tuple]): Vector of [n_samples, n_features] where n_samples is the number of samples and n_features is the number of features.
            labels (array-like [list, tuple, numpy.ndarray]): Vector of targets relative to examples for classification or regression.

        """

        # Fit classifier to data (examples, labels)
        try:
            self._classifier.fit(examples, labels)
        except Exception as error:
            raise Exception(error)
