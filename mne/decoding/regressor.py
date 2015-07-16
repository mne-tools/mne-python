# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np
from .base import BaseEstimator

class LinearRegressor(BaseEstimator):
    """
    This object clones a Linear Model from scikit-learn
    and updates the attribute for each fit. The model coefficient
    can be interpreted using the attribute patterns [1].

    Parameters
    ----------
    reg : object | None
        A linear regressor from scikit-learn with a fit method 
        that updates a coef_ attribute.
        If None the regressor will be a LinearRegressor

    Attributes
    ----------
    filters_ : ndarray
        If fit, the filters used to decompose the data, else None.
    patterns_ : ndarray
        If fit, the patterns used to restore M/EEG signals, else None.
    
    References
    ----------
    """
    def __init__(self, reg=None):
        if reg is None:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
        
        self.reg = reg
        self.patterns_ = None
        self.filters_ = None
    
    def fit(self, X, y):
        """Estimate the coeffiscient of the linear regressor.
        Save the coeffiscient in the attribute filters_ and 
        computes the attribute patterns_ using [1].

        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The data to estimate the coeffiscient.
        y : array, shape (n_epochs, n_target)
            The target for each epoch.

        Returns
        -------
        self : instance of LinearRegressor
            Returns the modified instance.
        
        References
        ----------
        """
        # fit the regressor
        self.reg.fit(X, y)
        assert hasattr(self.reg, 'coef_'), "reg needs a coef_ attribute to compute the patterns"
        # computes the patterns
        self.patterns_ = np.dot(X.T, np.dot(X, self.reg.coef_.T))
        self.filters_ = self.reg.coef_
        return self
    
    def predict(self, X):
        """Predict target variable for each trial in X.
        
        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The features for each epochs.
        
        Returns
        -------
        y_pred : array, shape (n_epochs, n_targets)
            Predicted target variables per epochs.
        """
        return self.reg.predict(X)
    
    def score(self, X, y):
        """
        Returns the score of the linear regressor computed
        on the given test data and classes.
        
        Parameters
        ----------
        X : array, shape (n_epochs, n_features)
            The data to transform.
        y : array, shape (n_epochs)
            The predictions for each epoch.

        Returns
        -------
        score : float
            Score of the linear regressor
        
        """
        return self.reg.score(X, y)
    
    
        