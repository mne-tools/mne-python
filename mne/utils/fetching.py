"""File downloading functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os


def _url_to_local_path(url, path):
    """Mirror a url path in a local destination (keeping folder structure)."""
    from urllib import parse, request

    destination = parse.urlparse(url).path
    # First char should be '/', and it needs to be discarded
    if len(destination) < 2 or destination[0] != "/":
        raise ValueError("Invalid URL")
    destination = os.path.join(path, request.url2pathname(destination)[1:])
    return destination
