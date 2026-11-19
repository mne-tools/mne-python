# Security Policy

## Supported Versions

New minor versions of MNE-Python are typically released twice per year.
Only the most current stable release is officially supported.
The unreleased, unstable "dev version" is also supported, though users
should beware that the API of the dev version is subject to change
without a proper 6-month deprecation cycle.

| Version | Supported                |
| ------- | ------------------------ |
| 1.14.x  | :heavy_check_mark: (dev) |
| 1.13.x  | :heavy_check_mark:       |
| < 1.13  | :x:                      |

## Reporting a Vulnerability

MNE-Python is software for analysis and visualization of brain activity
recorded with a variety of devices/modalities (EEG, MEG, ECoG, fNIRS, etc).
It is not expected that using MNE-Python will lead to security
vulnerabilities under normal use cases (i.e., running without administrator
privileges). However, if you think you have found a security vulnerability
in MNE-Python, **please do not report it as a GitHub issue**, in order to
keep the vulnerability confidential. Instead, please report it using
[GitHub's private vulnerability reporting](https://github.com/mne-tools/mne-python/security/advisories/new)
or by email to mne-core-dev-team@groups.io, and include a description and
proof-of-concept that is [short and self-contained](http://www.sscce.org/).

If the vulnerability is in a dependency (for example, h5io for HDF5 I/O),
please report it to that project as well.

Generally you will receive a response within one week. MNE-Python does not
award bounties for security vulnerabilities.
