# -*- coding: utf-8 -*-
"""File downloading functions."""
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import os
import shutil
import sys
import time
from urllib import parse, request

from .progressbar import ProgressBar
from .numerics import hashfunc
from .misc import sizeof_fmt
from ._logging import warn, logger, verbose


# Adapted from nilearn


def _get_http(url, temp_file_name, initial_size, file_size, timeout,
              verbose_bool):
    """Safely (resume a) download to a file from http(s)."""
    # Actually do the reading
    req = request.Request(url)
    if initial_size > 0:
        logger.debug('  Resuming at %s' % (initial_size,))
        req.add_header('Range', "bytes=%s-" % (initial_size,))
        try:
            response = request.urlopen(req, timeout=timeout)
            content_range = response.info().get('Content-Range')
            if (content_range is None or not content_range.startswith(
                    'bytes %s-' % (initial_size,))):
                raise IOError('Server does not support resuming')
        except Exception:
            # A wide number of errors can be raised here. HTTPError,
            # URLError... I prefer to catch them all and rerun without
            # resuming.
            return _get_http(
                url, temp_file_name, 0, file_size, timeout, verbose_bool)
    else:
        response = request.urlopen(req, timeout=timeout)
    total_size = int(response.headers.get('Content-Length', '1').strip())
    if initial_size > 0 and file_size == total_size:
        logger.info('Resuming download failed (resume file size '
                    'mismatch). Attempting to restart downloading the '
                    'entire file.')
        initial_size = 0
    total_size += initial_size
    if total_size != file_size:
        raise RuntimeError('URL could not be parsed properly '
                           '(total size %s != file size %s)'
                           % (total_size, file_size))
    mode = 'ab' if initial_size > 0 else 'wb'
    progress = ProgressBar(total_size, initial_value=initial_size,
                           spinner=True, mesg='file_sizes',
                           verbose_bool=verbose_bool)
    chunk_size = 8192  # 2 ** 13
    with open(temp_file_name, mode) as local_file:
        while True:
            t0 = time.time()
            chunk = response.read(chunk_size)
            dt = time.time() - t0
            if dt < 0.005:
                chunk_size *= 2
            elif dt > 0.1 and chunk_size > 8192:
                chunk_size = chunk_size // 2
            if not chunk:
                if verbose_bool:
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                break
            local_file.write(chunk)
            progress.update_with_increment_value(len(chunk),
                                                 mesg='file_sizes')


def _chunk_write(chunk, local_file, progress):
    """Write a chunk to file and update the progress bar."""
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
    try:
        # Check file size and displaying it alongside the download url
        # this loop is necessary to follow any redirects
        for _ in range(10):  # 10 really should be sufficient...
            u = request.urlopen(url, timeout=timeout)
            try:
                last_url, url = url, u.geturl()
                if url == last_url:
                    file_size = int(
                        u.headers.get('Content-Length', '1').strip())
                    break
            finally:
                u.close()
                del u
        else:
            raise RuntimeError('Too many redirects')
        logger.info('Downloading %s (%s)' % (url, sizeof_fmt(file_size)))

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
        # This should never happen if our functions work properly
        if initial_size > file_size:
            raise RuntimeError('Local file (%s) is larger than remote '
                               'file (%s), cannot resume download'
                               % (sizeof_fmt(initial_size),
                                  sizeof_fmt(file_size)))
        elif initial_size == file_size:
            # This should really only happen when a hash is wrong
            # during dev updating
            warn('Local file appears to be complete (file_size == '
                 'initial_size == %s)' % (file_size,))
        else:
            # Need to resume or start over
            scheme = parse.urlparse(url).scheme
            if scheme not in ('http', 'https'):
                raise NotImplementedError('Cannot use %s' % (scheme,))
            _get_http(url, temp_file_name, initial_size, file_size, timeout,
                      verbose_bool)

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
