"""Decoding and encoding, including machine learning and receptive fields."""

from .transformer import (PSDEstimator, Vectorizer,
                          UnsupervisedSpatialFilter, TemporalFilter,
                          Scaler, FilterEstimator)
from .mixin import TransformerMixin
from .base import BaseEstimator, LinearModel, get_coef, cross_val_multiscore
from .csp import CSP, SPoC
from .ems import compute_ems, EMS
from .time_frequency import TimeFrequency
from .receptive_field import ReceptiveField
from .time_delaying_ridge import TimeDelayingRidge
from .search_light import SlidingEstimator, GeneralizingEstimator
