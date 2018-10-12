# -*- coding: utf-8 -*-

"""
Demonstrates how to use NewsGroupsClassifier to fit a model to 
text data to classify newsgroup articles. This example comes 
from an example provided in the [Scikit-Learn documentation](http://
scikit-learn.org/stable/auto_examples/model_selection/
grid_search_text_feature_extraction.html). While the Scikit-Learn
example implements this example using a procedural approach, 
this implementation demonstrate a object-oriented programmming 
(OOP) approach.

Examples:

    Train the classifier and output a summary report:

        $ python train-classifier.py

    Run in interactive mode (to play around with objects and methods):

        $ python -i train-classifier.py

"""

from sklearn.datasets import fetch_20newsgroups

from NewsGroupsClassifier import NewsGroupsClassifier

if __name__ == "__main__":
    # Specify news groups categories
    categories = ["alt.atheism", "talk.religion.misc",]
    # Extract train data for news groups categories
    data = fetch_20newsgroups(subset="train", categories=categories)
    # Training examples
    train_examples = data.data
    # Training labels
    train_labels = data.target
    # Instantiate classifier
    clf = NewsGroupsClassifier()
    # Fit data
    clf.fit(train_examples, train_labels)
    # Report summary
    clf.summary