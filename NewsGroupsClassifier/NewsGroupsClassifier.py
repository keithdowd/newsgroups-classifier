# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

from .Classifier import Classifier
from .Metrics import Metrics


class NewsGroupsClassifier(Classifier, Metrics):
    """Customizable text classifier.

    NewsGroupsClassifier implements supervised text classification.
    It inherits functionality from Classifier, which provides the actual
    text classification implementation. This implementation depends on 
    the Scikit-Learn package. The NewsGroupsClassifier also inherits from
    Metrics, which provides access to a summary report (and other data)
    that describe fit outcomes.
    
    The NewsGroupsClassifier can be customized by passing (1) a Pipeline 
    object and (2) a parameters dict at instantiation. If one (or both) is 
    not provided, a default value is used. 

    Note that only the fitting (training) procedure is implement; no
    prediction implemention is provided.

    Args:
        pipeline (sklearn.pipeline.Pipeline): Pipeline object that 
            describes (1) feature extraction and (2) classification.
        parameters (dict): Dictionary that defines the parameter(s) and 
            values to search across for the Pipeline objects.

    """

    _class_name = "Newsgroups Classifier"

    def __init__(self, pipeline=None, parameters=None, *args, **kwargs):

        self._parameters = None
        self._pipeline = None

        # Set default values for parameters dict if not explicitly provided
        if parameters is None:
            self._parameters = {
                "vect__max_df": (0.5, 0.75, 1.0),
                # "vect__max_features": (None, 5000, 10000, 50000),
                # "vect__ngram_range": ((1, 1), (1, 2)),
                # "tfidf__norm": ('l1', 'l2'),
                # "tfidf__use_idf": (True, False),
                # "clf__alpha": (0.00001, 0.000001),
                "clf__max_iter": (5, 5),
                # "clf__n_iter": (10, 50, 80),
                # "clf__penalty": ('l2', 'elasticnet'),
                "clf__tol": (None, None),
            }

        # Set default pipeline dict if not explicity provided
        if pipeline is None:
            self._pipeline = Pipeline([
                ("vect", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("clf", SGDClassifier()),
            ])

        # Instantiate classifier
        Classifier.__init__(self, self._pipeline, self._parameters)

    def __str__(self, *args, **kwargs):
        """Human friendly readable label for class instance.

        Prints name of the classifier stored in _class_name if available 
        and prints "No classifier name is set" otherwise.

        Returns:
            str: Name of classifier.

        """

        if NewsGroupsClassifier._class_name == "" or NewsGroupsClassifier._class_name is None:
            return "No classifier name is set."

        return NewsGroupsClassifier._class_name
