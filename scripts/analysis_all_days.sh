#!/bin/bash

# Compute radiative profile features, conditional distributions of Qrad (net, lw, sw) on PW bins
# and draw figures for all EUREC4A days


days='20200122 20200124 20200126 20200128 20200131 20200202 20200205 20200207 20200209 20200211 20200213'
# days='20200126 '


for day in `echo $days`; do

    echo "--------------------------------------"
    echo "-------------- $day --------------"
    echo "--------------------------------------"
    echo
    echo "-- compute radiative features and distributions"
    echo
    python computeFeatures.py --day $day --overwrite True #(overwrite not yet implemented)

    echo
    echo "-- compute radiative scaling"
    echo
    python computeRadiativeScaling.py --day $day

done

echo
echo "DONE."
