#!/usr/bin/env bash
# test different set of margin and thresh_central parameters to find the best AP
for i in `seq 0 0.01 0.2`;
    do
    for j in `seq 0 0.05 0.95`
        do
            echo margin: $i, thresh_central: $j
            /home/binghao/software/anaconda2/bin/python ./evaluate_from_outputs.py --margin $i --overlap_thresh_central $j
        done
    done
