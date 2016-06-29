"""A collection of classes and functions for fitting neural encoding models."""
from .model import EventRelatedRegressor
from .feature import (DataDelayer, EventsBinarizer,
                      DataSubsetter, clean_inputs)
