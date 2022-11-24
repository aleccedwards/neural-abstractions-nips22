#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 
./$model-synth.sh
./$model-spaceex.sh



