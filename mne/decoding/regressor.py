# Authors: Romain Trachel <trachelr@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: BSD (3-clause)

import numpy as np


class LinearRegressor():
    """
    This object clones a Linear Model from sklearn
    and updates the attribute for each fit. The model coefficient
    can be interpreted using the attribute patterns [1].

    Parameters
    ----------
    reg : object | None
        A linear regressor from sklearn with a fit method 
        that updates a coef_ attribute.
        If None the classifier will be a LinearRegressor

    Attributes
    ----------
    filters_ : ndarray
        If fit, the filters used to decompose the data, else None.
    patterns_ : ndarray
        If fit, the patterns used to restore M/EEG signals, else None.
    
    References
    ----------
    """
    def __init__(self, reg):
        if reg is None:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression()
        
        self.reg = reg
    
    def fit(self, X, y):
        """Estimate the coeffiscient of the linear regressor.
        Save the coeffiscient in the attribute filters_ and 
        computes the attribute patterns_ using [1].

        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The data to estimate the coeffiscient.
        y : array, shape=(n_epochs, n_target)
            The target for each epoch.

        Returns
        -------
        self : instance of LinearRegressor
            Returns the modified instance.
        
        References
        ----------
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        if not isinstance(y, np.ndarray):
            raise ValueError("y should be of type ndarray (got %s)."
                             % type(y))
        # check for features dimension
        X = np.atleast_2d(X)
        if len(X.shape) != 2:
            raise ValueError("X dimension should be 2 (n_epochs x n_features)"
                             " instead of ", X.shape)
        # fit the regressor
        self.reg.fit(X, y)
        # computes the patterns
        if hasattr(self.reg, 'coef_'):
            self.patterns_ = np.dot(X.T, np.dot(X, self.reg.coef_.T))
            self.filters_ = self.reg.coef_
        return self
    
    def transform(self, X, y=None):
        """Transform the data using the linear regressor.

        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The data to transform.
        y : array, shape=(n_epochs)
            The class for each epoch.

        Returns
        -------
        y_pred : array, shape=(n_epochs)
            Predicted target per epoch.
        
        """
        return self.predict(X)
    
    def fit_transform(self, X, y):
        """fit the data and transform it using the linear regressor.

        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The data to transform.
        y : array, shape=(n_epochs)
            The class for each epoch.

        Returns
        -------
        y_pred : array, shape=(n_epochs)
            Predicted target per epoch.
        
        """
        return self.fit(X, y).predict(X)
    
    def predict(self, X):
        """Predict target variable for each trial in X.
        
        Parameters
        ----------
        X : array, shape=(n_epochs, n_features)
            The features for each epochs.
        
        Returns
        -------
        y_pred : array, shape=(n_epochs, n_targets)
            Predicted target variables per epochs.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be of type ndarray (got %s)."
                             % type(X))
        # check for features dimension
        X = np.atleast_2d(X)
        if len(X.shape) != 2:
            raise ValueError("X dimension should be 2 (n_epochs x n_features)"
                             " instead of ", X.shape)
        
        y_pred = self.clf.predict(X)
        return y_pred
    
        