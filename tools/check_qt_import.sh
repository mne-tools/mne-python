#!/bin/bash -ef

set -eo pipefail

if [[ "$1" == "" ]]; then
    echo "Qt library must be provided as first argument"
    exit 1
fi
echo "Checking if Python-Qt binding $1 is available"
echo ""
set -x
LD_DEBUG=libs python -c "from $1.QtWidgets import QApplication, QWidget; app = QApplication([])"
set +x
echo ""
echo "Python-Qt binding $1 is available!"
