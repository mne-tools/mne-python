"""
Introduction: dealing with artifacts
====================================

Since FieldTrip supports the data of many different acquisition systems, the
particular artifacts in your data might behave very different from the
artifata. Therefore you should be aware of the different approaches and of
the variability of artifact rejection (automatic/manual) procedures described
onwards. At the end consider always to visual inspect your data,
rejection.
"""

###############################################################################
# Background: what is an artifact?
# --------------------------------

###############################################################################
# How does MNE manage artifacts?
# ------------------------------

# MNE deals with artifacts by first identifying them, and subsequently
# removing them....
