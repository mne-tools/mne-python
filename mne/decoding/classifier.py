# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from .base import BaseEstimator

class LinearClassifier(BaseEstimator):
    """
    This object clones a Linear Classifier from scikit-learn
    and updates the attribute for each fit. The model coefficient
    can be interpreted using the attribute patterns [1].

    Parameters
    ----------
    clf : object | None
        A linear classifier from scikit-learn with a fit method 
        that updates a coef_ attribute.
        If None the classifier will be a LogisticRegression

    Attributes
    ----------
    filters_ : ndarray
        If fit, the filters used to decompose the data, else None.
    patterns_ : ndarray
        If fit, the patterns used to restore M/EEG signals, else None.
    
    References
    ----------
    """
    def __init__(self, clf=None):
        if clf is None:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression()
        
        self.clf = clf
        self.patterns_ = None
        self.filters_ = None
    
    def fit(self, X, y):
        """Estimate the coeffiscient of the linear classifier.
        Save the coeffiscient in the attribute filters_ and 
        computes the attribute patterns_ using [1].

        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The data to estimate the coeffiscient.
        y : array, shape (n_epochs)
            The class for each epoch.

        Returns
        -------
        self : instance of LinearClassifier
            Returns the modified instance.
        
        References
        ----------
        """
        # fit the classifier
        self.clf.fit(X, y)
        # computes the patterns
        assert hasattr(self.clf, 'coef_'), "clf needs a coef_ attribute to compute the patterns"
        self.patterns_ = np.dot(X.T, np.dot(X, self.clf.coef_.T))
        self.filters_ = self.clf.coef_
        
        return self
    
    def transform(self, X, y=None):
        """Transform the data using the linear classifier.

        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The data to transform.
        y : array, shape (n_epochs)
            The class for each epoch.

        Returns
        -------
        y_pred : array, shape (n_epochs)
            Predicted class label per epoch.
        
        """
        return self.clf.transform(X)
    
    
    def fit_transform(self, X, y):
        """fit the data and transform it using the linear classifier.

        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The data to transform.
        y : array, shape (n_epochs)
            The class for each epoch.

        Returns
        -------
        y_pred : array, shape (n_epochs)
            Predicted class label per epoch.
        
        """
        return self.fit(X, y).transform(X)
    
    def predict(self, X, y=None):
        """Predict class labels for each trial in X.
        
        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The features of each trial to predict class label.
        
        Returns
        -------
        y_pred : array, shape (n_epochs)
            Predicted class label per epoch.
        """
        return self.clf.predict(X)
        
    
        