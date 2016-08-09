"""A collection of classes and functions for fitting encoding models."""
from .feature import (FeatureDelayer, EventsBinarizer, delay_time_series,
                      binarize_events, DelaysVectorizer)
from .model import SampleMasker, get_coefs
