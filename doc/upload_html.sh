#!/usr/bin/env bash

#scp -r build/html/* martinos-data:/web/html/mne/
rsync -rltvz --delete --perms --chmod=g+w build/html/ martinos-data:/web/html/mne/ -essh
ssh martinos-data "chgrp -R megweb /web/html/mne"
