#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 
time ./$model-experiment-na.sh

 
