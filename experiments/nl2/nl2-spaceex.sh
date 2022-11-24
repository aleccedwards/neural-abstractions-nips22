#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/} 

/spaceex_exe/spaceex --model-file nl2-model.xml --rel-err '1.0E-12' --abs-err '1.0E-15' --output-error '0' --scenario 'stc' --directions 'oct' --set-aggregation 'chull' --verbosity m --time-horizon '5' --sampling-time '1' --simu-init-sampling-points '0' --flowpipe-tolerance '0.01' --flowpipe-tolerance-rel '0' --iter-max '-1' --initially 'x0<=0.025 & x0>=-0.025 & x1>=-0.9 & x1 <= -0.85 & u0==0 & u1==0 & t==0 ' --forbidden 'x0 <=0.05 & x0 >= -0.05 & x1 >= -0.8 & x1 <=-0.7' --system 'NA' --output-format 'TXT' --output-variables x0,x1 --output-file spaceex.gen


