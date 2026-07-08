import os
import numpy as np

sats = [
    'cfosat',  # O, don
    'cryosat-2',  # 1 ,done
    'envisat',  # 2, done
    'ers-1',  # 3, done
    'ers-2',  # 4, done
    'gfo',  # 5, done
    'jason-1',  # 6, done
    'jason-2',  # 7, done
    'jason-3',  # 8, done
    'saral',  # 9, done
    'sentinel-3_a',  # 10,  done
    'sentinel-3_b',  # 11, done
    'sentinel-6_a',  # 12, done
    'swot',  # 13, done
    'topex-poseidon_topex',  # 14, done
    'topex-poseidon_poseidon',  # 15, done
]

inds = np.array([4, 15], dtype=int)
for sat in range(0, 1):  # ,16): # inds:
    os.system('qsub -v "Y=' + str(sat) + '" detect_one_sat.pbs')
