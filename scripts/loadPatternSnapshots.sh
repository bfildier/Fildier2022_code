#!/bin/bash

## Load GOES 2.5km res images, 1 image per day, for the selection of days analyzed

hour='14'
min='00'
time="${hour}:${min}"

# EUREC4A_movie_DIR='/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_movies'
repo_DIR='/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization'

days='20200122 20200124 20200126 20200128 20200131 20200202 20200205 20200207 20200209 20200211 20200213'

for day in `echo $days`; do

    echo "--------------------------------------"
    echo "-------------- $day --------------"
    echo "--------------------------------------"

# COMMENTED LINES DOWNLOAD THE DESIRED IMAGES -- ASK AUTHOR FOR CODE
#
#    cd ${EUREC4A_movie_DIR}
#    echo "load GOES snapshots in PNG"
#    python ./scripts/make_images_opendap.py -d $day --start_time $time --stop_time $time

#    echo "move to local repo"
#    mv ./images/GOES16__${day}_${hour}${min}.png ${repo_DIR}/images/patterns/PNG/

#    cd -

    echo "Add grid, day, time and HALO circle"
    python showPatternSnapshot.py --day ${day} --time ${hour}${min} 

done

echo
echo "DONE."

cd -
