import os
import numpy as np

sats = [
    'cfosat',
    'cryosat-2',
    'envisat',
    'ers-1',
    'ers-2',
    'gfo',
    'jason-1',
    'jason-2',
    'jason-3',
    'saral',
    'sentinel-3_a',
    'sentinel-3_b',
    'sentinel-6_a',
    'swot',
    'topex-poseidon_topex',
    'topex-poseidon_poseidon',
]
for sat in sats:
    os.system('qsub -v "Y=' + sat + '"detect_one_sat.pbs')
