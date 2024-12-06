Fix bug where invalid data types (e.g., ``np.ndarray``s) could be used in some
:class:`mne.io.Info` fields like ``info["subject_info"]["weight"]``, by `Eric Larson`_.