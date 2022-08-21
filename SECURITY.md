# Security Policy

## Supported Versions

New minor versions of MNE-Python are typically released twice per year.
Only the most current stable release is officially supported.
The unreleased, unstable "dev version" is also supported, though users
should beware that the API of the dev version is subject to change
without a proper 6-month deprecation cycle.

| Version | Supported                |
| ------- | ------------------------ |
| 1.2.x   | :heavy_check_mark: (dev) |
| 1.1.x   | :heavy_check_mark:       |
| < 1.1   | :x:                      |

## Reporting a Vulnerability

MNE-Python is software for analysis and visualization of brain activity
recorded with a variety of devices/modalities (EEG, MEG, ECoG, fNIRS, etc).
It is not expected that using MNE-Python will lead to security
vulnerabilities under normal use cases (i.e., running without administrator
privileges). However, if you think you have found a security vulnerability
in MNE-Python, **please do not report it as a GitHub issue**, in order to 
keep the vulnerability confidential. Instead, please report it to
mne-core-dev-team@groups.io and include a description and proof-of-concept
that is [short and self-contained](http://www.sscce.org/).

Generally you will receive a response within one week. MNE-Python does not
award bounties for security vulnerabilities.
