"""A collection of classes and functions for fitting neural encoding models."""
from .feature import (FeatureDelayer, EventsBinarizer)
from .model import (SampleMasker, get_coefs,
                    get_final_est, remove_outliers)
