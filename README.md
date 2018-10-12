# News Groups Classifier
Example of refactoring a procedural approach for feature extraction on a text classification problem to an object-oriented (OOP) approach for solving the same problem. More specifically, the code in this repo demonsrates an OOP approach applied to a text classification problem for classifying posts made by users to newsgroups.The code for the procedural approach is provided by the [`Scikit-Learn`](http://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html) library. The newsgroups data is provided by the (now classic) 20 newsgroups dataset.

## Setup
This code was built and tested with [`Python v3.7.0`](https://www.python.org/ftp/python/3.7.0/python-3.7.0.exe) and [`pipenv v2018.10.9`](https://github.com/pypa/pipenv). The instructions for running this code assume `pipenv` is the tool used for spinning up virtual environments, but `pip` works great, too, so just adapt the instructions as necessary.

### Dependencies
* [`NumPy`](http://www.numpy.org/): v1.5.2 or higher
* [`Scikit-Learn`](http://scikit-learn.org/): v0.20.0 or higher
* [`SciPy`](https://www.scipy.org/): v1.1.0 or higher

### Run
The quickest way to get running is to `git clone` and `cd` into this repo's root directory and execute the following at the command line. 

**Install dependencies**
```console
$ pipenv install
```

**Activate virtual environment**
```console
$ pipenv shell
```

**Run script**
```console
$ python train-classifier.py
```

### Explore
To more easily explore the output from the classifier run the script in Python's interactive mode with the following command:

```console
$ python -i train-classifier.py
```

### Fit Classifier on Other Data
The following code demonstrates how easy it is to train on other data.

```python
# import NewsGroupsClassifier
from NewsGroupsClassifiers import NewsGroupsClassifier

# pass in pipeline object, parameter dict to customize fit
clf = NewsGroupsClassifier() 

# train_examples, train_labels are text training data
clf.fit(train_examples, train_labels) 

# view summary report
clf.summary
```

## Future Improvements
* Encapsulate training data in a data object and pass to classifier
* Better error and exception handling (log exceptions and stack traces to file on disk)
* Abstract feature extraction pipeline components into own object (use `FeatureUnion` to combine extraction and classifier pipeline components)

## Authors
* Keith Dowd <[keith.dowd at gmail dot com](mailto:keith.dowd@gmail.com)>

## Acknowledgements
* Thank you kindly for allowing me the opportunity to participate in this code challenge! :smile: