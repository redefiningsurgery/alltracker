#!/bin/bash
TARFILE="monkey.mp4"
echo "downloading ${TARFILE} from dropbox"
wget --max-redirect=20 -O ${TARFILE} https://www.dropbox.com/scl/fi/fm2m3ylhzmqae05bzwm8q/monkey.mp4?rlkey=ibf81gaqpxkh334rccu7zrioe&st=mli9bqb6&dl=1
