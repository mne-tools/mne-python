#!/bin/bash -e

brew install python@3.10 ffmpeg
which python
python -c "import sys; assert sys.version.startswith('3.10')"
which pip
