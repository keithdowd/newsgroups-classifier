# -*- coding: utf-8 -*-

from pprint import pprint


class Metrics:
    """Model fit metrics.

    Metrics implements an easy-to-use API for extracting model
    fit metrics. Metrics is intended to be used in conjuction with
    (inherited by) a classifier with these fit metrics as 
    attributes.

    """

    @property
    def best_estimator(self, *args, **kwargs):
        """Best estimator found by the classifier fitting process.

        The best estimator found by the fitting process, where "best" 
        is defined as the estimator with the highest validation 
        accuracy score.

        Returns:
            sklearn.pipeline.Pipeline: Best estimator identified by
                the fitting process.

        Raises:
            AttributeError: If best_estimator_ is not an attribute on 
                _classifier.

        """

        try:
            return self._classifier.best_estimator_
        except AttributeError:
            raise AttributeError(
                "No best estimator attribute is available for this classifier.")

    @property
    def best_params(self, *args, **kwargs):
        """Best set of parameters found by the classifier fitting process.

        The best set of parameters and values found by the fitting process,
        where "best" is defined as the estimator with the highest 
        validation accuracy score.

        Returns:
            dict: Best set of parameters and values for estimator.

        Raises:
            AttributeError: If best_estimator_ (and thus get_params()) is 
                not an attribute on _classifier.

        """

        try:
            return self._classifier.best_estimator_.get_params()
        except AttributeError:
            raise AttributeError(
                "No best parameters is/are available for this classifier.")

    @property
    def best_score(self, *args, **kwargs):
        """Best score found by the classifier fitting process.

        The best classification accuracy score found by the fitting 
        process, where "best" is defined as the estimator with the highest 
        validation accuracy score.

        Returns:
            float: Best classification accuracy score.
        
        Raises:
            AttributeError: If best_score_ is not an attribute on 
                _classifier.

        """

        try:
            return self._classifier.best_score_
        except AttributeError:
            raise AttributeError(
                "No best score is available for this classifier.")

    @property
    def summary(self, *args, **kwargs):
        """Classification summary report.

        Human-readable and easily interpretable summary report that describes
        the results of the fitting process, with a focus on the
        characteristics of the best estimator.

        Report is printed to stdout.

        Returns:
            None

        """

        print("\nCLASSIFICATION SUMMARY REPORT:")
        print("=" * 30)
        print("\nPipeline:\n\t", [step for step, _ in self._pipeline.steps])
        print()
        print("Parameters:")
        for param_name in sorted(self._parameters.keys()):
            print(f"\t{param_name} : {self._parameters[param_name]}")
        print()
        print(f"Best score:\n\t{self.best_score}")
        print()
        print("Best parameters set:")
        for param_name in sorted(self._parameters.keys()):
            print(f"\t{param_name} : {self.best_params[param_name]}")
        print()
