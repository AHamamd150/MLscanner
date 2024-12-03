#!/bin/bash
if [ -d "work/" ]; then
  rm -rf work
fi  

[ $# -eq 0 ] && { echo "No input given, please run ./run.sh --help for more information"; exit 1; }
if [ $1 == '--help' ]; then
  cat "./source/help"
  exit 0
fi
cp -rf scan_input.py source
python3 source/MLs_HEP.py $1
