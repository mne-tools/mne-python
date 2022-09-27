#!/bin/bash -e

if ! -f /opt/homebrew/opt/python@3.10/bin/python; then
    brew install python@3.10
    sudo ln -s python3 /opt/homebrew/opt/python@3.10/bin/python
    sudo ln -s pip3 /opt/homebrew/opt/python@3.10/bin/pip
fi
which python
python -c "import sys; assert sys.version.startswith('3.10')"
which pip
