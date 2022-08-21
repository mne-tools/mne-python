# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
import os.path as op

from ..utils import _manifest_check_download
from ...utils import verbose, get_subjects_dir, _check_option, _validate_type

_AGES = '2wk 1mo 2mo 3mo 4.5mo 6mo 7.5mo 9mo 10.5mo 12mo 15mo 18mo 2yr'
# https://github.com/christian-oreilly/infant_template_paper/releases
_ORIGINAL_URL = 'https://github.com/christian-oreilly/infant_template_paper/releases/download/v0.1-alpha/{subject}.zip'  # noqa: E501
# Formatted the same way as md5sum *.zip on Ubuntu:
_ORIGINAL_HASHES = """
851737d5f8f246883f2aef9819c6ec29  ANTS10-5Months3T.zip
32ab6d025f4311433a82e81374f1a045  ANTS1-0Months3T.zip
48ef349e7cc542fdf63ff36d7958ab57  ANTS12-0Months3T.zip
bba22c95aa97988c6e8892d6169ed317  ANTS15-0Months3T.zip
e1bfe5e3ef380592822ced446a4008c7  ANTS18-0Months3T.zip
fa7bee6c0985b9cd15ba53820cd72ccd  ANTS2-0Months3T.zip
2ad90540cdf42837c09f8ce829458a35  ANTS2-0Weeks3T.zip
73e6a8b2579b7959a96f7d294ffb7393  ANTS2-0Years3T.zip
cb7b9752894e16a4938ddfe220f6286a  ANTS3-0Months3T.zip
16b2a6804c7d5443cfba2ad6f7d4ac6a  ANTS4-5Months3T.zip
dbdf2a9976121f2b106da96775690da3  ANTS6-0Months3T.zip
75fe37a1bc80ed6793a8abb47681d5ab  ANTS7-5Months3T.zip
790f7dba0a264262e6c1c2dfdf216215  ANTS9-0Months3T.zip
"""
_MANIFEST_PATH = op.dirname(__file__)


@verbose
def fetch_infant_template(age, subjects_dir=None, *, verbose=None):
    """Fetch and update an infant MRI template.

    Parameters
    ----------
    age : str
        Age to download. Can be one of ``{'2wk', '1mo', '2mo', '3mo', '4.5mo',
        '6mo', '7.5mo', '9mo', '10.5mo', '12mo', '15mo', '18mo', '2yr'}``.
    subjects_dir : str | None
        The path to download the template data to.
    %(verbose)s

    Returns
    -------
    subject : str
        The standard subject name, e.g. ``ANTS4-5Month3T``.

    Notes
    -----
    If you use these templates in your work, please cite
    :footcite:`OReillyEtAl2021` and :footcite:`RichardsEtAl2016`.

    .. versionadded:: 0.23

    References
    ----------
    .. footbibliography::
    """
    # Code used to create the lists:
    #
    # $ for name in 2-0Weeks 1-0Months 2-0Months 3-0Months 4-5Months 6-0Months 7-5Months 9-0Months 10-5Months 12-0Months 15-0Months 18-0Months 2-0Years; do wget https://github.com/christian-oreilly/infant_template_paper/releases/download/v0.1-alpha/ANTS${name}3T.zip; done  # noqa: E501
    # $ md5sum ANTS*.zip
    # $ python
    # >>> import os.path as op
    # >>> import zipfile
    # >>> names = [f'ANTS{name}3T' for name in '2-0Weeks 1-0Months 2-0Months 3-0Months 4-5Months 6-0Months 7-5Months 9-0Months 10-5Months 12-0Months 15-0Months 18-0Months 2-0Years'.split()]  # noqa: E501
    # >>> for name in names:
    # ...     with zipfile.ZipFile(f'{name}.zip', 'r') as zip:
    # ...         names = sorted(name for name in zip.namelist() if not zipfile.Path(zip, name).is_dir())  # noqa: E501
    # ...     with open(f'{name}.txt', 'w') as fid:
    # ...         fid.write('\n'.join(names))
    _validate_type(age, str, 'age')
    _check_option('age', age, _AGES.split())
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    subjects_dir = op.abspath(subjects_dir)
    unit = dict(wk='Weeks', mo='Months', yr='Years')[age[-2:]]
    first = age[:-2].split('.')[0]
    dash = '-5' if '.5' in age else '-0'
    subject = f'ANTS{first}{dash}{unit}3T'
    # Actually get and create the files
    subj_dir = op.join(subjects_dir, subject)
    os.makedirs(subj_dir, exist_ok=True)
    # .zip -> hash mapping
    orig_hashes = dict(line.strip().split()[::-1]
                       for line in _ORIGINAL_HASHES.strip().splitlines())
    _manifest_check_download(
        manifest_path=op.join(_MANIFEST_PATH, f'{subject}.txt'),
        destination=subj_dir,
        url=_ORIGINAL_URL.format(subject=subject),
        hash_=orig_hashes[f'{subject}.zip'],
    )
    return subject
