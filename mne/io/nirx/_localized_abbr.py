"""Localizations for meas_date extraction."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# This file was generated on 2021/01/31 on an Ubuntu system.
# When getting "unsupported locale setting" on Ubuntu (e.g., with localepurge),
# use "sudo locale-gen de_DE" etc. then "sudo update-locale".

"""
import datetime
import locale
print('_localized_abbr = {')
for loc in ('en_US.utf8', 'de_DE', 'fr_FR', 'it_IT'):
    print(f'    {repr(loc)}: {{')
    print('        "month": {', end='')
    month_abbr = set()
    for month in range(1, 13):  # Month as locale’s abbreviated name
        locale.setlocale(locale.LC_TIME, "en_US.utf8")
        dt = datetime.datetime(year=2000, month=month, day=1)
        val = dt.strftime("%b").lower()
        locale.setlocale(locale.LC_TIME, loc)
        key = dt.strftime("%b").lower()
        month_abbr.add(key)
        print(f'{repr(key)}: {repr(val)}, ', end='')
    print('},  # noqa')
    print('        "weekday": {', end='')
    weekday_abbr = set()
    for day in range(1, 8):  # Weekday as locale’s abbreviated name.
        locale.setlocale(locale.LC_TIME, "en_US.utf8")
        dt = datetime.datetime(year=2000, month=1, day=day)
        val = dt.strftime("%a").lower()
        locale.setlocale(locale.LC_TIME, loc)
        key = dt.strftime("%a").lower()
        assert key not in weekday_abbr, key
        weekday_abbr.add(key)
        print(f'{repr(key)}: {repr(val)}, ', end='')
    print('},  # noqa')
    print('    },')
print('}\n')
"""

# TODO: this should really be outsourced to a dedicated module like arrow or babel
_localized_abbr = {
    "en_US.utf8": {
        "month": {
            "jan": "jan",
            "feb": "feb",
            "mar": "mar",
            "apr": "apr",
            "may": "may",
            "jun": "jun",
            "jul": "jul",
            "aug": "aug",
            "sep": "sep",
            "oct": "oct",
            "nov": "nov",
            "dec": "dec",
        },  # noqa
        "weekday": {
            "sat": "sat",
            "sun": "sun",
            "mon": "mon",
            "tue": "tue",
            "wed": "wed",
            "thu": "thu",
            "fri": "fri",
        },  # noqa
    },
    "de_DE": {
        "month": {
            "jan": "jan",
            "feb": "feb",
            "mär": "mar",
            "apr": "apr",
            "mai": "may",
            "jun": "jun",
            "jul": "jul",
            "aug": "aug",
            "sep": "sep",
            "okt": "oct",
            "nov": "nov",
            "dez": "dec",
        },  # noqa
        "weekday": {
            "sa": "sat",
            "so": "sun",
            "mo": "mon",
            "di": "tue",
            "mi": "wed",
            "do": "thu",
            "fr": "fri",
        },  # noqa
    },
    "fr_FR": {
        "month": {
            "janv.": "jan",
            "févr.": "feb",
            "mars": "mar",
            "avril": "apr",
            "mai": "may",
            "juin": "jun",
            "juil.": "jul",
            "août": "aug",
            "sept.": "sep",
            "oct.": "oct",
            "nov.": "nov",
            "déc.": "dec",
        },  # noqa
        "weekday": {
            "sam.": "sat",
            "dim.": "sun",
            "lun.": "mon",
            "mar.": "tue",
            "mer.": "wed",
            "jeu.": "thu",
            "ven.": "fri",
        },  # noqa
    },
    "it_IT": {
        "month": {
            "gen": "jan",
            "feb": "feb",
            "mar": "mar",
            "apr": "apr",
            "mag": "may",
            "giu": "jun",
            "lug": "jul",
            "ago": "aug",
            "set": "sep",
            "ott": "oct",
            "nov": "nov",
            "dic": "dec",
        },  # noqa
        "weekday": {
            "sab": "sat",
            "dom": "sun",
            "lun": "mon",
            "mar": "tue",
            "mer": "wed",
            "gio": "thu",
            "ven": "fri",
        },  # noqa
    },
}
