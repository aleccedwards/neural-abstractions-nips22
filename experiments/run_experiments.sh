#!/usr/bin/env bash

bench=("nl1" "nl2" "jet" "steam" "exp" "watertank")
cd nl1
for b in "${bench[@]}"
do
    cd ../$b
    ./table-1-row.sh
done
