"""A collection of classes and functions for fitting neural encoding models."""
from .model import EncodingModel
from .feature import (DataDelayer, EventsBinarizer,
                      DataSubsetter, delay_timeseries)
