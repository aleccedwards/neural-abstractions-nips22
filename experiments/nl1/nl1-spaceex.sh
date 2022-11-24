#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 

/spaceex_exe/spaceex --model-file nl1-model.xml --rel-err '1.0E-12' --abs-err '1.0E-15' --output-error '0' --scenario 'stc' --directions 'oct' --set-aggregation 'none' --verbosity m --time-horizon '1.5' --sampling-time '1' --simu-init-sampling-points '0' --flowpipe-tolerance '0.01' --flowpipe-tolerance-rel '0' --iter-max '-1' --initially 'x0>= 0 & x0 <= 0.05 & x1 >= 0 & x1 <= 0.1 & u0==0 & u1==0 & t==0' --forbidden 'x0>=0.35 & x0 <=0.45 & x1>=0.1 & x1<=0.2' --system 'NA' --output-format 'TXT' --output-variables x0,x1 --output-file spaceex.gen

