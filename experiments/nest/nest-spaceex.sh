#!/usr/bin/env bash
 
/spaceex_exe/spaceex --model-file exp-model.xml --rel-err '1.0E-12' --abs-err '1.0E-15' --output-error '0' --scenario 'stc' --directions 'oct' --set-aggregation 'chull' --verbosity m --time-horizon '0.5' --sampling-time '1' --simu-init-sampling-points '0' --flowpipe-tolerance '0.005' --flowpipe-tolerance-rel '0' --iter-max '-1' --initially 'x0>=0.45 & x0<=0.5 & x1>=0.86 & x1 <= 0.91 & u0==0 & u1==0 & t==0 ' --forbidden 'x0 >= 0.3 & x0 <=0.4 & x1 >=0.5 & x1 <=0.6' --system 'NA' --output-format 'TXT' --output-variables x0,x1 --output-file spaceex.gen
