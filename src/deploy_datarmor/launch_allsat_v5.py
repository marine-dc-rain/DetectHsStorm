import os
import numpy as np

sats = [
    'cfosat', # don
    'cryosat-2', # done
    'envisat', # done
    'ers-1', #done
    'ers-2',#done
    'gfo', #done
    'jason-1',#done
    'jason-2',#done
    'jason-3',#done
    'saral',#done
    'sentinel-3_a',#done
    'sentinel-3_b',#done
    'sentinel-6_a',#done
    'swot',#done
    'topex-poseidon_topex',#done
    'topex-poseidon_poseidon',#done
]
for sat, _ in enumerate(sats):
    os.system('qsub -v "Y=' + str(sat) + '"detect_one_sat.pbs')
