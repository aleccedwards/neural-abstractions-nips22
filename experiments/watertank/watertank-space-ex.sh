#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 

/spaceex_exe/spaceex --model-file watertank-model.xml --rel-err '1.0E-12' --abs-err '1.0E-15' --output-error '0' --scenario 'stc' --directions 'oct' --set-aggregation 'chull' --verbosity m --time-horizon '5' --sampling-time '1' --simu-init-sampling-points '0' --flowpipe-tolerance '0.01' --flowpipe-tolerance-rel '0' --iter-max '-1' --initially 'x0<=0.01 & x0 >= 0.0 & u0==0 & t==0 ' --forbidden 'x0 >= 2.0' --system 'NA' --output-format 'TXT' --output-variables x0,t --output-file spaceex.gen

