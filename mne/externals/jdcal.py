# -*- coding: utf-8 -*-
"""Functions for converting between Julian dates and calendar dates.

A function for converting Gregorian calendar dates to Julian dates, and
another function for converting Julian calendar dates to Julian dates
are defined. Two functions for the reverse calculations are also
defined.

Different regions of the world switched to Gregorian calendar from
Julian calendar on different dates. Having separate functions for Julian
and Gregorian calendars allow maximum flexibility in choosing the
relevant calendar.

All the above functions are "proleptic". This means that they work for
dates on which the concerned calendar is not valid. For example,
Gregorian calendar was not used prior to around October 1582.

Julian dates are stored in two floating point numbers (double).  Julian
dates, and Modified Julian dates, are large numbers. If only one number
is used, then the precision of the time stored is limited. Using two
numbers, time can be split in a manner that will allow maximum
precision. For example, the first number could be the Julian date for
the beginning of a day and the second number could be the fractional
day. Calculations that need the latter part can now work with maximum
precision.

A function to test if a given Gregorian calendar year is a leap year is
defined.

Zero point of Modified Julian Date (MJD) and the MJD of 2000/1/1
12:00:00 are also given.

This module is based on the TPM C library, by Jeffery W. Percival. The
idea for splitting Julian date into two floating point numbers was
inspired by the IAU SOFA C library.

:author: Prasanth Nair
:contact: prasanthhn@gmail.com
:license: BSD (http://www.opensource.org/licenses/bsd-license.php)

NB: Code has been heavily adapted for streamlined use by mne-python devs
"""


import numpy as np

MJD_0 = 2400000


def ipart(x):
    """Return integer part of given number."""
    return np.modf(x)[1]


def jcal2jd(year, month, day):
    """Julian calendar date to Julian date.

    The input and output are for the proleptic Julian calendar,
    i.e., no consideration of historical usage of the calendar is
    made.

    Parameters
    ----------
    year : int
        Year as an integer.
    month : int
        Month as an integer.
    day : int
        Day as an integer.

    Returns
    -------
    jd: int
        Julian date.
    """
    year = int(year)
    month = int(month)
    day = int(day)

    jd = 367 * year
    x = ipart((month - 9) / 7.0)
    jd -= ipart((7 * (year + 5001 + x)) / 4.0)
    jd += ipart((275 * month) / 9.0)
    jd += day
    jd += 1729777
    return jd


def jd2jcal(jd):
    """Julian calendar date for the given Julian date.

    The input and output are for the proleptic Julian calendar,
    i.e., no consideration of historical usage of the calendar is
    made.

    Parameters
    ----------
    jd: int
        The Julian date.

    Returns
    -------
    y, m, d: int, int, int
        Three element tuple containing year, month, day.
    """
    j = jd + 1402
    k = ipart((j - 1) / 1461.0)
    l = j - (1461.0 * k)
    n = ipart((l - 1) / 365.0) - ipart(l / 1461.0)
    i = l - (365.0 * n) + 30.0
    j = ipart((80.0 * i) / 2447.0)
    day = i - ipart((2447.0 * j) / 80.0)
    i = ipart(j / 11.0)
    month = j + 2 - (12.0 * i)
    year = (4 * k) + n + i - 4716.0
    return int(year), int(month), int(day)
