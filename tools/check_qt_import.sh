#!/bin/bash -ef

if [[ "$1" == "" ]]; then
    echo "Qt library must be provided as first argument"
    exit 1
fi
LD_DEBUG=libs python -c "from $1.QtWidgets import QApplication, QWidget; app = QApplication([])"
echo "Python-Qt binding $1 is available"
