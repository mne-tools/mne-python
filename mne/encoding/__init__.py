"""A collection of classes and functions for fitting neural encoding models."""
from .model import EncodingModel, EventRelatedRegressor
from .feature import (DataDelayer, EventsBinarizer,
                      DataSubsetter)
