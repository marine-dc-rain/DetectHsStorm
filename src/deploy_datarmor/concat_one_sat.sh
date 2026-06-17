#!/bin/bash

source /home1/datahome/mdecarlo/.bashrc
micromamba activate storms_env
python3 /home1/datahome/mdecarlo/TEMPETES/DetectHsStorm/src/detect_storms_in_sat.py -o l2p -s 1 -y 1991 -m 8 -Y 2025 -M 12 --multiprocessing -n $Y >& /$SCRATCH/concat_storms_sat_${PBS_JOBID}