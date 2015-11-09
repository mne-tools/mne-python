#!/usr/bin/env bash

#scp -r build/html/* martinos-data:/web/html/mne/
rsync -rltvz --delete --perms --chmod=g+w _build/html/ martinos-data:/web/html/ext/mne/stable -essh
ssh martinos-data "chgrp -R megweb /web/html/ext/mne/stable"
