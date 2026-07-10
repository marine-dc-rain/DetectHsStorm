#!/bin/bash

source /home1/datahome/mdecarlo/.bashrc
micromamba activate storms_env
python3 /home1/datahome/mdecarlo/TEMPETES/DetectHsStorm/src/detect_storms_in_sat.py -o l2p -y 2019 -m 7 -Y 2025 -M 12 -r --multiprocessing -n $Y >& /$SCRATCH/detect_storms_sat_${PBS_JOBID}
# python3 /home1/datahome/mdecarlo/TEMPETES/DetectHsStorm/src/detect_storms_in_sat.py -o l2p -y 1991 -m 8 -Y 2025 -M 12 -r --multiprocessing -n $Y >& /$SCRATCH/detect_storms_sat_${PBS_JOBID}