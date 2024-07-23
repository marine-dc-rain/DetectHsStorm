import numpy as np

Nb_CPU = 1
isWW3 = 1
if isWW3==1:
	# PATH = '/home/datawork-WW3/HINDCAST/GLOBMULTI_ERA5_GLOBCUR_01/GLOB-30M/'
	PATH = '/home/mdecarlo/Documents/PROJETS/SWOT/Model'
	# FORMAT_IN = 'YYYY/FIELD_NC/LOPS_WW3-GLOB-30M_YYYYMM.nc'
	# PATH = '/home/mdecarlo/Documents/code/data_test/'
	# FORMAT_IN = 'Model_WW3_hs-GLOB-30M_YYYYMM.nc'
	# FORMAT_OUT_detect = 'WW3_detect_storm_YYYY_MM.nc'
	FORMAT_IN = 'SWOT_WW3-GLOB-30M_YYYYMM.nc'
	FORMAT_OUT_detect = 'WW3_detect_storm_YYYY_MM.nc'
	FORMAT_OUT_tracking = 'WW3_tracking_storm_YYYY_MM.nc'
	# PATH_SAVE_detect = '/home1/datawork/mdecarlo/TBH_Model/WW3_LOPS/detect/'
	# PATH_SAVE_tracking = '/home1/datawork/mdecarlo/TBH_Model/WW3_LOPS/tracking/'
	# PATH_SAVE_detect = '/home/mdecarlo/Documents/code/StormsDetection/output/detect'
	# PATH_SAVE_tracking = '/home/mdecarlo/Documents/code/StormsDetection/output/tracking'
	PATH_SAVE_detect = '/home/mdecarlo/Documents/PROJETS/TBH_Tempetes_bdd_historique/Storms_src/output/detect'
	PATH_SAVE_tracking = '/home/mdecarlo/Documents/PROJETS/TBH_Tempetes_bdd_historique/Storms_src/output/tracking'
else:
	#PATH = '/home/mdecarlo/Documents/DATA/'
	PATH = '/dataref/ecmwf/intranet/ERA5/'#
	FORMAT_IN = 'YYYY/MM/era_5-copernicus__YYYYMMDD.nc'
	FORMAT_OUT_detect = 'ERA5_detect_storm_YYYY_MM.nc'
	FORMAT_OUT_tracking = 'ERA5_tracking_storm_YYYY_MM.nc'
	PATH_SAVE_detect = '/home1/datawork/mdecarlo/TBH_Model/ERA5/detect/'
	PATH_SAVE_tracking = '/home1/datawork/mdecarlo/TBH_Model/ERA5/tracking/'

# filedist2coast = '/home1/datahome/mdecarlo/Python_functions/distance2coast180.nc'
# filedist2coast = '/home/mdecarlo/Documents/code/data_test/distance2coast180.nc'
filedist2coast = '/home/mdecarlo/Documents/PROJETS/codes_Python/distance2coast180.nc'
swh_crit_max = 16.
dswh_crit = .5
min_swh = 4.
swh_levels = np.arange(min_swh, swh_crit_max+dswh_crit, dswh_crit)
swh_levels.sort()
levels = swh_levels[::-1]
Npix_min = 16
amp_thresh = 1./5#

d_thresh_min = 200. # min linear dimension of storm [km]
d_thresh_max = 6000. # max linear dimension of storm [km]
area_forgotten_ratio = 0.2
min_area = 500

threshold_dist = 400
