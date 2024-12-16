#!/bin/bash

set -e
set -o pipefail

./tools/setup_xvfb.sh
sudo apt install -qq graphviz optipng python3.12-venv python3-venv libxft2 ffmpeg
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
python3.12 -m venv ~/python_env
echo "set -e" >> $BASH_ENV
echo "set -o pipefail" >> $BASH_ENV
echo "export XDG_RUNTIME_DIR=/tmp/runtime-circleci" >> $BASH_ENV
echo "export MNE_FULL_DATE=true" >> $BASH_ENV
source tools/get_minimal_commands.sh
echo "export MNE_3D_BACKEND=pyvistaqt" >> $BASH_ENV
echo "export MNE_BROWSER_BACKEND=qt" >> $BASH_ENV
echo "export MNE_BROWSER_PRECOMPUTE=false" >> $BASH_ENV
echo "export MNE_ADD_CONTRIBUTOR_IMAGE=true" >> $BASH_ENV
echo "export MNE_REQUIRE_RELATED_SOFTWARE_INSTALLED=true" >> $BASH_ENV
echo "export PATH=~/.local/bin/:$PATH" >> $BASH_ENV
echo "export DISPLAY=:99" >> $BASH_ENV
echo "source ~/python_env/bin/activate" >> $BASH_ENV
mkdir -p ~/.local/bin
ln -s ~/python_env/bin/python ~/.local/bin/python
echo "BASH_ENV:"
cat $BASH_ENV
mkdir -p ~/mne_data
touch pattern.txt
