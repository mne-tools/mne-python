# -*- coding: utf-8 -*-
"""File downloading functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os
import shutil
import time
from urllib import parse, request
from urllib.error import HTTPError, URLError

from .progressbar import ProgressBar
from .numerics import hashfunc
from .misc import sizeof_fmt
from ._logging import logger, verbose


# Adapted from nilearn


def _get_http(url, temp_file_name, initial_size, timeout, verbose_bool):
    """Safely (resume a) download to a file from http(s)."""
    # Actually do the reading
    response = None
    extra = ''
    if initial_size > 0:
        logger.debug('  Resuming at %s' % (initial_size,))
        req = request.Request(
            url, headers={'Range': 'bytes=%s-' % (initial_size,)})
        try:
            response = request.urlopen(req, timeout=timeout)
            content_range = response.info().get('Content-Range', None)
            if (content_range is None or not content_range.startswith(
                    'bytes %s-' % (initial_size,))):
                raise IOError('Server does not support resuming')
        except (KeyError, HTTPError, URLError, IOError):
            initial_size = 0
            response = None
        else:
            extra = ', resuming at %s' % (sizeof_fmt(initial_size),)
    if response is None:
        response = request.urlopen(request.Request(url), timeout=timeout)
    file_size = int(response.headers.get('Content-Length', '0').strip())
    file_size += initial_size
    url = response.geturl()
    logger.info('Downloading %s (%s%s)' % (url, sizeof_fmt(file_size), extra))
    del url
    mode = 'ab' if initial_size > 0 else 'wb'
    progress = ProgressBar(file_size, initial_size, unit='B',
                           mesg='Downloading', unit_scale=True,
                           unit_divisor=1024)
    del file_size
    chunk_size = 8192  # 2 ** 13
    with open(temp_file_name, mode) as local_file:
        while True:
            t0 = time.time()
            chunk = response.read(chunk_size)
            dt = time.time() - t0
            if dt < 0.01:
                chunk_size *= 2
            elif dt > 0.1 and chunk_size > 8192:
                chunk_size = chunk_size // 2
            if not chunk:
                break
            local_file.write(chunk)
            progress.update_with_increment_value(len(chunk))


@verbose
def _fetch_file(url, file_name, print_destination=True, resume=True,
                hash_=None, timeout=30., hash_type='md5', verbose=None):
    """Load requested file, downloading it if needed or requested.

    Parameters
    ----------
    url: string
        The url of file to be downloaded.
    file_name: string
        Name, along with the path, of where downloaded file will be saved.
    print_destination: bool, optional
        If true, destination of where file was saved will be printed after
        download finishes.
    resume: bool, optional
        If true, try to resume partially downloaded files.
    hash_ : str | None
        The hash of the file to check. If None, no checking is
        performed.
    timeout : float
        The URL open timeout.
    hash_type : str
        The type of hashing to use such as "md5" or "sha1"
    %(verbose)s
    """
    # Adapted from NISL:
    # https://github.com/nisl/tutorial/blob/master/nisl/datasets.py
    if hash_ is not None and (not isinstance(hash_, str) or
                              len(hash_) != 32) and hash_type == 'md5':
        raise ValueError('Bad hash value given, should be a 32-character '
                         'string:\n%s' % (hash_,))
    temp_file_name = file_name + ".part"
    verbose_bool = (logger.level <= 20)  # 20 is info
    scheme = parse.urlparse(url).scheme
    if scheme not in ('http', 'https'):
        raise NotImplementedError('Cannot use scheme %r' % (scheme,))
    try:
        # Triage resume
        if not os.path.exists(temp_file_name):
            resume = False
        if resume:
            with open(temp_file_name, 'rb', buffering=0) as local_file:
                local_file.seek(0, 2)
                initial_size = local_file.tell()
            del local_file
        else:
            initial_size = 0
        _get_http(url, temp_file_name, initial_size, timeout, verbose_bool)

        # check hash sum eg md5sum
        if hash_ is not None:
            logger.info('Verifying hash %s.' % (hash_,))
            hashsum = hashfunc(temp_file_name, hash_type=hash_type)
            if hash_ != hashsum:
                raise RuntimeError('Hash mismatch for downloaded file %s, '
                                   'expected %s but got %s'
                                   % (temp_file_name, hash_, hashsum))
        shutil.move(temp_file_name, file_name)
        if print_destination is True:
            logger.info('File saved as %s.\n' % file_name)
    except Exception:
        logger.error('Error while fetching file %s.'
                     ' Dataset fetching aborted.' % url)
        raise


def _url_to_local_path(url, path):
    """Mirror a url path in a local destination (keeping folder structure)."""
    destination = parse.urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != '/':
        raise ValueError('Invalid URL')
    destination = os.path.join(path, request.url2pathname(destination)[1:])
    return destination
