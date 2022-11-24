#!/usr/bin/env bash
model=${PWD##*/}
model=${model:-/}

declare -A XI

XI[jet]="[[0.45, -0.6], [0.45, -0.55], [0.5, -0.6], [0.5, -0.55]]"
XI[nl2]="[[-0.025, -0.9], [-0.025, -0.85], [0.025, -0.9], [0.025, -0.85]]"
XI[nl1]="[[0, 0], [0, 0.1], [0.05, 0.1], [0.05, 0]]"
XI[steam]="[[0.7, -0.05, 0.7],[0.7, -0.05, 0.75],[0.7, 0.05, 0.7],[0.7, 0.05, 0.75],[0.75, -0.05, 0.7],[0.75, -0.05, 0.75],[0.75, 0.05, 0.7],[0.75, 0.05, 0.75]]" 
XI[exp]="[[0.45, 0.86], [0.45, 0.91], [0.5, 0.86], [0.5, 0.91]]"
XI[watertank]="[[0], [0.01]]"
cd nl1
for b in "${!XI[@]}"; do
    cd ../$b
    python3 ../../main.py -c /neural-abstraction/experiments/${b}/${b}-config.yaml --initial "${XI[$b]}"
    sed '/^ *initially/s/"$/ \& loc()==Init"/' ${b}-spaceex.cfg > temp.cfg
    sed -i '/output-variables/c\output-variables = "x0" ' temp.cfg
    java -jar /mnt/DATA/Documents/Git-repos/HyST/src/Hyst.jar -output ${b}-abs-flowstar.model -input ${b}-model_init.xml temp.cfg -tool flowstar "" -verbose
    rm temp.cfg
    rm ${b}-model_init.xml
done

