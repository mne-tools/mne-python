"""Decoding and encoding, including machine learning and receptive fields."""
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "base": ["BaseEstimator", "LinearModel", "cross_val_multiscore", "get_coef"],
        "csp": ["CSP", "SPoC"],
        "ems": ["EMS", "compute_ems"],
        "mixin": ["TransformerMixin"],
        "receptive_field": ["ReceptiveField"],
        "search_light": ["GeneralizingEstimator", "SlidingEstimator"],
        "ssd": ["SSD"],
        "time_delaying_ridge": ["TimeDelayingRidge"],
        "time_frequency": ["TimeFrequency"],
        "transformer": [
            "FilterEstimator",
            "PSDEstimator",
            "Scaler",
            "TemporalFilter",
            "UnsupervisedSpatialFilter",
            "Vectorizer",
        ],
    },
)
