#!/usr/bin/env bash

num_procs=30
num_jobs="\j"

widths=("10" "10 10" "15 15")
benchmarks=("jet")

for benchmark in "${benchmarks[@]}"
do
    for WIDTH in "${widths[@]}"
    do
        for s in {0..9}
        do
            while (( ${num_jobs@P} >= num_procs )); do
                wait -n
            done
            python3 main.py -b $benchmark -w $WIDTH -f subresults-seed$s-width${WIDTH// /_}-benchmark$benchmark -c experiments/scoping-exp.yaml --seed $s & 
        done
    done
done

widths=("10" "20")
benchmarks=("steam")

for benchmark in "${benchmarks[@]}"
do
    for WIDTH in "${widths[@]}"
    do
        for s in {0..9}
        do
            while (( ${num_jobs@P} >= num_procs )); do
                wait -n
            done
            python3 main.py -b $benchmark -w $WIDTH -f subresults-seed$s-width${WIDTH// /_}-benchmark$benchmark -c experiments/scoping-exp.yaml --seed $s & 
        done
    done
done

widths=("10" "20" "20 20")
benchmarks=("exp")
for benchmark in "${benchmarks[@]}"
do
    for WIDTH in "${widths[@]}"
    do
        for s in {0..9}
        do
            while (( ${num_jobs@P} >= num_procs )); do
                wait -n
            done
            python3 main.py -b $benchmark -w $WIDTH -f subresults-seed$s-width${WIDTH// /_}-benchmark$benchmark -c experiments/scoping-exp.yaml --seed $s & 
        done
    done
done
wait
awk '(NR == 1) || (FNR > 1)' results/subresults*.csv >> results/results.csv
find results/subresults*.csv -delete
