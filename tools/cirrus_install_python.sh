#!/bin/bash -e

if [[ ! -f /opt/homebrew/opt/python@3.10/bin/python ]]; then
    brew install python@3.10
fi
which python
python -c "import sys; assert sys.version.startswith('3.10')"
which pip
