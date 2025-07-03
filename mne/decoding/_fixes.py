# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    from sklearn.utils.validation import check_array, check_X_y

    # Use a limited version pulled from sklearn 1.7
    def validate_data(
        _estimator,
        /,
        X="no_validation",
        y="no_validation",
        reset=True,
        validate_separately=False,
        skip_check_array=False,
        **check_params,
    ):
        """Validate input data and set or check feature names and counts of the input.

        This helper function should be used in an estimator that requires input
        validation. This mutates the estimator and sets the `n_features_in_` and
        `feature_names_in_` attributes if `reset=True`.

        .. versionadded:: 1.6

        Parameters
        ----------
        _estimator : estimator instance
            The estimator to validate the input for.

        X : {array-like, sparse matrix, dataframe} of shape \
                (n_samples, n_features), default='no validation'
            The input samples.
            If `'no_validation'`, no validation is performed on `X`. This is
            useful for meta-estimator which can delegate input validation to
            their underlying estimator(s). In that case `y` must be passed and
            the only accepted `check_params` are `multi_output` and
            `y_numeric`.

        y : array-like of shape (n_samples,), default='no_validation'
            The targets.

            - If `None`, :func:`~sklearn.utils.check_array` is called on `X`. If
            the estimator's `requires_y` tag is True, then an error will be raised.
            - If `'no_validation'`, :func:`~sklearn.utils.check_array` is called
            on `X` and the estimator's `requires_y` tag is ignored. This is a default
            placeholder and is never meant to be explicitly set. In that case `X` must
            be passed.
            - Otherwise, only `y` with `_check_y` or both `X` and `y` are checked with
            either :func:`~sklearn.utils.check_array` or
            :func:`~sklearn.utils.check_X_y` depending on `validate_separately`.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.
            If False, the input will be checked for consistency with data
            provided when reset was last True.

            .. note::

            It is recommended to call `reset=True` in `fit` and in the first
            call to `partial_fit`. All other methods that validate `X`
            should set `reset=False`.

        validate_separately : False or tuple of dicts, default=False
            Only used if `y` is not `None`.
            If `False`, call :func:`~sklearn.utils.check_X_y`. Else, it must be a tuple
            of kwargs to be used for calling :func:`~sklearn.utils.check_array` on `X`
            and `y` respectively.

            `estimator=self` is automatically added to these dicts to generate
            more informative error message in case of invalid input data.

        skip_check_array : bool, default=False
            If `True`, `X` and `y` are unchanged and only `feature_names_in_` and
            `n_features_in_` are checked. Otherwise, :func:`~sklearn.utils.check_array`
            is called on `X` and `y`.

        **check_params : kwargs
            Parameters passed to :func:`~sklearn.utils.check_array` or
            :func:`~sklearn.utils.check_X_y`. Ignored if validate_separately
            is not False.

            `estimator=self` is automatically added to these params to generate
            more informative error message in case of invalid input data.

        Returns
        -------
        out : {ndarray, sparse matrix} or tuple of these
            The validated input. A tuple is returned if both `X` and `y` are
            validated.
        """
        no_val_X = isinstance(X, str) and X == "no_validation"
        no_val_y = y is None or (isinstance(y, str) and y == "no_validation")

        if no_val_X and no_val_y:
            raise ValueError("Validation should be done on X, y or both.")

        default_check_params = {"estimator": _estimator}
        check_params = {**default_check_params, **check_params}

        if skip_check_array:
            if not no_val_X and no_val_y:
                out = X
            elif no_val_X and not no_val_y:
                out = y
            else:
                out = X, y
        elif not no_val_X and no_val_y:
            out = check_array(X, input_name="X", **check_params)
        elif no_val_X and not no_val_y:
            out = check_array(y, input_name="y", **check_params)
        else:
            if validate_separately:
                # We need this because some estimators validate X and y
                # separately, and in general, separately calling check_array()
                # on X and y isn't equivalent to just calling check_X_y()
                # :(
                check_X_params, check_y_params = validate_separately
                if "estimator" not in check_X_params:
                    check_X_params = {**default_check_params, **check_X_params}
                X = check_array(X, input_name="X", **check_X_params)
                if "estimator" not in check_y_params:
                    check_y_params = {**default_check_params, **check_y_params}
                y = check_array(y, input_name="y", **check_y_params)
            else:
                X, y = check_X_y(X, y, **check_params)
            out = X, y

        return out
