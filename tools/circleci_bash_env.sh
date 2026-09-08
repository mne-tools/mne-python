#!/bin/bash

set -e
set -o pipefail

# The regional EC2 mirror started 503ing on 2026-09-04 and apt spins on it for ~10 min before
# giving up, so fall back to the canonical one; self-healing, and removable once it is reliable
if ! curl -fsS --max-time 10 -o /dev/null "http://us-east-1.ec2.archive.ubuntu.com/ubuntu/dists/$(lsb_release -cs)/Release"; then
    echo "Regional apt mirror is unhealthy, falling back to archive.ubuntu.com"
    sudo sed -i 's|//[a-z0-9-]*\.ec2\.archive\.ubuntu\.com|//archive.ubuntu.com|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources /etc/apt/sources.list.d/*.list 2>/dev/null || true
fi
# timeout: setup_xvfb.sh runs apt itself, so it can stall the same way the calls below can
curl -fsSL https://raw.githubusercontent.com/mne-tools/mne-tools/main/tools/setup_xvfb.sh | timeout 600 bash
# Need different installs for 24.04 and 26.04
if [[ $(lsb_release -rs) == "26.04" ]]; then
    EXTRA_DEPS="libgvplugin-neato-layout8"
else
    EXTRA_DEPS=""
fi
APT_OPTS="-o Acquire::Retries=3 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30"
# no -qq: it hides download progress, which trips CircleCI's 10 min no-output timeout (gh-14103)
# timeout: that progress output also defeats the no-output timeout, so a stalled mirror otherwise
# spins for an hour and buries the real error under CircleCI's 50 MB output cap
timeout 600 sudo apt install -y $APT_OPTS graphviz optipng python3-venv libxft2 ffmpeg libtirpc-dev $EXTRA_DEPS
# r-base-dev rather than r-base: rpy2-rinterface has no manylinux wheel so it builds against
# R's headers, but we don't need r-recommended (the r-cran-* set) or the html manuals
timeout 600 sudo apt install -y $APT_OPTS r-base-dev
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
timeout 600 sudo apt install -y $APT_OPTS ./google-chrome-stable_current_amd64.deb
python -m venv ~/python_env
echo "set -e" >> $BASH_ENV
echo "set -o pipefail" >> $BASH_ENV
echo "export XDG_RUNTIME_DIR=/tmp/runtime-circleci" >> $BASH_ENV
echo "export MNE_FULL_DATE=true" >> $BASH_ENV
echo "export MNE_3D_BACKEND=pyvistaqt" >> $BASH_ENV
echo "export MNE_QT_BACKEND=PySide6" >> $BASH_ENV
echo "export MNE_BROWSER_BACKEND=qt" >> $BASH_ENV
echo "export MNE_BROWSER_PRECOMPUTE=false" >> $BASH_ENV
echo "export MNE_ADD_CONTRIBUTOR_IMAGE=true" >> $BASH_ENV
echo "export MNE_REQUIRE_RELATED_SOFTWARE_INSTALLED=true" >> $BASH_ENV
# Persist numba's on-disk JIT cache across runs, with generic CPU to make the
# cached objects portable
echo "export NUMBA_CACHE_DIR=$HOME/.cache/mne-numba" >> $BASH_ENV
echo "export NUMBA_CPU_NAME=generic" >> $BASH_ENV
echo "export NUMBA_CPU_FEATURES=" >> $BASH_ENV
echo "export PATH=~/.local/bin/:$PATH" >> $BASH_ENV
echo "export DISPLAY=:99" >> $BASH_ENV
echo "source ~/python_env/bin/activate" >> $BASH_ENV
mkdir -p ~/.local/bin
ln -s ~/python_env/bin/python ~/.local/bin/python
echo "BASH_ENV:"
cat $BASH_ENV
mkdir -p ~/mne_data
# Must exist before save_cache runs, even if nothing ever compiles into it
mkdir -p ~/.cache/mne-numba
touch pattern.txt
