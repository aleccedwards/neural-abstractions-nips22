#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 

./$model-synth.sh
./$model-space-ex.sh

