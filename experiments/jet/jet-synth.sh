#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 
python3 ../../main.py -c $model-config.yaml
