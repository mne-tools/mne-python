"""A collection of classes and functions for fitting encoding models."""
from .feature import FeatureDelayer, EventsBinarizer, delay_timeseries
from .model import SampleMasker, get_coefs, get_final_est
