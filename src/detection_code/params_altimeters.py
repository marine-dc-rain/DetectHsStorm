# -*- coding: utf-8 -*-
"""
Created on Sat Dec. 21 2024

@author: M. Passaro & F. ardhuin
"""

"""altimeters_parameters.py: A Python module for LRM waveform models
"""
# ================================================================
# Imports
# ----------------------------------------------------------------
import numpy as np
import netCDF4
from netCDF4 import Dataset
import xarray as xr

FORMAT_OUT_detalt = 'ALT_detect_storm_YYYY_MM.nc'
FORMAT_OUT_detalt_summary = 'ALT_detect_storm_T1_T2_TYPE.nc'
PATH_SAVE_det_sat = '/home1/datahome/mdecarlo/TEMPETES/sat_extract_v5'
PATH_SAVE_summary_sat = '/home/datawork-WW3/PROJECT/CCI/STORMS/v5'

hs_thresh = 9.0
hs_thresh_min = 5.0
min_length = 10


addvarlist = [
    'era5_wind_eastward',
    'era5_wind_northward',
    'era5_surface_pressure',
    'distance_to_coast',
    'bathymetry',
    'ww3_wave_skewness',
    'ww3_mean_wave_period',
    'ww3_wavenumber_peakdness',
    'ww3_emb',
    'ww3_swh',
    'ww3_mean_wave_period_t0m1',
    'ww3_mean_wave_direction',
    'ww3_peak_wave_period',
]

var_supplist = [
    'swh_with_8m_offset_correction',
    'swh_with_8m_offset_correction_rms',
    'swh_with_8m_offset_correction_numval',
    'swh_with_8m_offset_correction_quality_level',
    'swh_with_8m_offset_correction_rejection_flags',
]


# l2paddvars = [
#             'ww3_swh',
#             'ww3_wavenumber_peakdness',
#             denoisevar,
#             denoiseunc,
#             'era5_swh',
#             'swh_rms',
#             'swh_numval',
#         ]
######################  Defines parameters
def alti_paths_GDR(mission):

    PATH_ALTI_in = ''
    PATH_ALTI_ii = ''

    if mission.lower() in ['cryosat2']:
        PATH_ALTI_in = ' /home/datawork-cersat-public/provider/esa/satellite/l2/cryosat-2/siral/sir_lrm_l2/version_e/data/date/YYYY/???/CS_LTA__SIR_LRM_2__YYYYMM*.nc'
        PATH_ALTI_in = ' /home/datawork-cersat-public/provider/esa/satellite/l2/cryosat-2/siral/sir_lrm_l2/version_e/data/date/YYYY/*/*2__YYYYMM*.nc'
        TAG_ALTI = 'GDR_CS2'

    if mission.lower() in ['ers1']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/esa/satellite/l2/ers-1/ra/esa-reaper/ers_alt_2s/data/date/YYYY/???/E1_REAP_ERS_ALT_2S_YYYYMM*.NC'
        TAG_ALTI = 'GDR_ER1'

    if mission.lower() in ['ers2']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/esa/satellite/l2/ers-2/ra/esa-reaper/ers_alt_2s/data/date/YYYY/???/E2_REAP_ERS_ALT_2S_YYYYMM*.NC'
        TAG_ALTI = 'GDR_ER2'

    if mission.lower() in ['envisat']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/esa/satellite/l2/envisat/ra2/sgdr/v3/YYYY/???/ENV_RA_2_MWS____YYYYMM*.nc'
        TAG_ALTI = 'GDR_ENV'

    if mission.lower() in ['tp-topex']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/topex-poseidon/topex/gdr/version_f/YYYY/*/TP_GPN_2PfP???_???_YYYYMM*.nc'
        TAG_ALTI = 'GDR_TPT'

    if mission.lower() in ['jason1']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-1/poseidon-2/sgdr/version_e/data/date/YYYY/*/JA1_GPS_2PeP???_???_YYYYMM*.nc'
        TAG_ALTI = 'GDR_JA1'

    if mission.lower() in ['jason2']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-2/poseidon-3/sgdr/version_d/data/date/YYYY/*/JA2_GPS_*_YYYYMM*.nc'
        TAG_ALTI = 'GDR_JA2'

    if mission.lower() in ['jason3f']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-3/poseidon-3b/sgdr/version_f/data/date/YYYY/*/JA3_GPS_*_YYYYMM??_*.nc'
        PATH_ALTI_ii = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-3/poseidon-3b/igdr/version_f/data/date/YYYY/*/JA3_IPN_*_YYYYMM??_*.nc'
        TAG_ALTI = 'GDR_JA3'

    if mission.lower() in ['sentinel3a']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-3a/sral/s3a_sr_2_wat____o_st/YYYY/???/S3A_SR_2_WAT____YYYYMM*.SEN3/standard_measurement.nc'
        TAG_ALTI = 'GDR_S3A'

    if mission.lower() in ['sentinel3b']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-3b/sral/s3b_sr_2_wat____o_st/YYYY/???/S3B_SR_2_WAT____YYYYMM*.SEN3/standard_measurement.nc'
        TAG_ALTI = 'GDR_S3B'

    if mission.lower() in ['sentinel6']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-6a/poseidon-4/p4_2__lr/f08/YYYY/???/S6A_P4_2__LR______YYYYMM*.SEN6/S6A_P4_2__LR_STD__NT_???_???_YYYYMM*.nc'
        PATH_ALTI_ii = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-6a/poseidon-4/p4_2__lr/f09/YYYY/???/S6A_P4_2__LR______YYYYMM*.SEN6/S6A_P4_2__LR_STD__NT_???_???_YYYYMM*.nc'
        TAG_ALTI = 'GDR_S6A'

    if mission.lower() in ['swot']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/swot/poseidon-3c/igdr/YYYY/???/SWOT_IPS_2PfP???_???_YYYYMM*.nc'
        PATH_ALTI_ii = '/home/datawork-cersat-public/provider/aviso/satellite/l2/swot/poseidon-3c/igdr/2.0/YYYY/???/SWOT_IPS_2PfP???_???_YYYYMM*.nc'
        TAG_ALTI = 'GDR_SWO'

    if mission.lower() in ['cfosat']:
        PATH_ALTI_in = (
            '/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op??/*/YYYY/*/*F_YYYYMM*'
        )
        TAG_ALTI = 'GDR_CFO'

    if mission.lower() in ['hy2a']:
        PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/hy-2a/alt/rs-igdr/data/YYYY/???/H2A_RA1_RDR_2PT_????_????_YYYYMM??_*.nc'
        TAG_ALTI = 'GDR_H2A'

    return PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI, []


######################  Defines paths for CCI files (l2)
def alti_paths_cci(mission, origin_type='1hz', version='v5'):
    print('Setting file paths for mission:', mission.lower())
    PATH_ALTI_in = ''
    PATH_ALTI_ii = ''
    if origin_type == 'l2p':
        if version == 'v5':
            rootpath = (
                '/home/datawork-cersat-public/provider/cci_seastate/products/v5/data/satellite/altimeter/l2p-swh/'
            )
            # rootpath = '/home/mdecarlo/Documents/Projets/CCI_SeaState/DATA/v5/'
            tag1 = 'CCI_l2p_v5_'
        else:  # 'v4'
            # rootpath='/home/datawork-cersat-public/cache/project/cciseastate/data/v4/altimeter/l2p/'
            rootpath = '/home/ref-cersat-public/ocean-waves/cci-seastate/v4/data/satellite/altimeter/l2p/'
            tag1 = 'CCI_l2p_v4_'
    else:
        rootpath = '/home/datawork-cersat-public/cache/project/cciseastate/data/v4/altimeter/1hz/'
        tag1 = 'CCIv4_'
    if mission.lower() in ['saral']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-SARAL-'
        else:
            tag2 = 'CS_LTA__SIR_LRM_1B_'
        PATH_ALTI_in = rootpath + 'saral/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_ii = ''  # rootpath + 'saral/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'SAR'
        addvarlist_saved = addvarlist

    if mission.lower() in ['cryosat2']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-CryoSat-2-'
            PATH_ALTI_in = rootpath + 'cryosat-2/YYYY/???/' + tag2 + 'YYYYMM*.nc'
            PATH_ALTI_ii = ''  # rootpath + 'cryosat-2/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        else:
            tag2 = 'CS_LTA__SIR_LRM_1B_'
            PATH_ALTI_in = rootpath + 'cryosat-2e/YYYY/???/' + tag2 + 'YYYYMM*.nc'
            PATH_ALTI_ii = rootpath + 'cryosat-2e/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'CS2'
        addvarlist_saved = addvarlist

    if mission.lower() in ['cfosat']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-CFOSAT-'
            PATH_ALTI_in = rootpath + 'cfosat/YYYY/???/' + tag2 + 'YYYYMM*.nc'
            PATH_ALTI_ii = ''  # rootpath + 'cryosat-2/YYYY/???/' + tag2 + 'YYYYMM*.nc'

        TAG_ALTI = tag1 + 'CFO'
        addvarlist_saved = addvarlist

    if mission.lower() in ['ers1']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-ERS-1-'
        else:
            tag2 = 'E1_REAP_ERS_ALT_2__'
            PATH_ALTI_in = rootpath + 'ers-1-reaper/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_in = rootpath + 'ers-1/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'ER1'
        addvarlist_saved = addvarlist + var_supplist

    if mission.lower() in ['ers2']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-ERS-2-'
        else:
            tag2 = 'E2_REAP_ERS_ALT_2__'
            PATH_ALTI_in = rootpath + 'ers-2-reaper/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_in = rootpath + 'ers-2/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'ER2'
        addvarlist_saved = addvarlist + var_supplist

    if mission.lower() in ['envisat']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Envisat-'
        else:
            tag2 = 'ENV_RA_2_MWS____'
            PATH_ALTI_in = rootpath + 'envisat-v3/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_in = rootpath + 'envisat/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'ENV'
        addvarlist_saved = addvarlist + var_supplist

    if mission.lower() in ['gfo']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-GFO-'
            PATH_ALTI_in = rootpath + 'gfo/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'GFO'
        addvarlist_saved = addvarlist

    if mission.lower() in ['tp-topex']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Topex-Poseidon-'
        else:
            tag2 = 'TP_GPN_2P?????_???_'
            PATH_ALTI_in = rootpath + 'topexf_topex_a/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_in = rootpath + 'topex-poseidon_topex/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_ii = ""  # rootpath + 'topex-poseidon/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'TPT'
        addvarlist_saved = addvarlist

    if mission.lower() in ['tp-poseidon']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Topex-Poseidon-'
        else:
            tag2 = 'TP_GPN_2P?????_???_'
            PATH_ALTI_in = rootpath + 'topexf_topex_a/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_in = rootpath + 'topex-poseidon_poseidon/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_ii = ""  # rootpath + 'topex-poseidon/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'TPP'
        addvarlist_saved = addvarlist

    if mission.lower() in ['jason1']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Jason-1-'
        else:
            tag2 = 'JA1_GPS_2P?????_???_'
        PATH_ALTI_in = rootpath + 'jason-1/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'JA1'
        addvarlist_saved = addvarlist + var_supplist
    if mission.lower() in ['jason2']:
        # PATH_ALTI_in = '/home/datawork-cersat-public/provider/cci_seastate/products/v3/data/satellite/altimeter/l2_20Hz/l2/jason-2/YYYY/????/JA2_???_2P?????_???_YYYYMM*.nc'
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Jason-2-'
        else:
            tag2 = 'JA2_GPS_2P?????_???_'
        #         PATH_ALTI_in = rootpath+'jason-2d/YYYY/???/'+tag2+'YYYYMM*.nc'
        PATH_ALTI_in = rootpath + 'jason-2/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'JA2'
        addvarlist_saved = addvarlist + var_supplist

    if mission.lower() in ['jason3']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Jason-3-'
        else:
            tag2 = 'JA3_GPS_2P?????_???_'
        PATH_ALTI_in = rootpath + 'jason-3/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_ii = ""  # rootpath + 'jason-3d/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'JA3'
        addvarlist_saved = addvarlist + var_supplist

    if mission.lower() in ['sentinel3a']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Sentinel-3_A-'
        else:
            tag2 = 'S3A_P4_2__LR______'
        PATH_ALTI_in = rootpath + 'sentinel-3_a/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'S3A'
        addvarlist_saved = addvarlist

    if mission.lower() in ['sentinel3b']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Sentinel-3_B-'
        else:
            tag2 = 'S3B_P4_2__LR______'
        PATH_ALTI_in = rootpath + 'sentinel-3_b/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'S3B'
        addvarlist_saved = addvarlist

    if mission.lower() in ['sentinel6a']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-Sentinel-6_A-'
        else:
            tag2 = 'S6A_P4_2__LR______'
        PATH_ALTI_in = rootpath + 'sentinel-6_a/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        PATH_ALTI_ii = ""  # rootpath + 'sentinel-6/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'S6A'
        addvarlist_saved = addvarlist + var_supplist

    if mission.lower() in ['swot']:
        if origin_type == 'l2p':
            tag2 = 'ESACCI-SEASTATE-L2P-SWH-SWOT-'
        PATH_ALTI_in = rootpath + 'swot/YYYY/???/' + tag2 + 'YYYYMM*.nc'
        TAG_ALTI = tag1 + 'SWO'
        addvarlist_saved = addvarlist + var_supplist

    print('path :', PATH_ALTI_in)
    return PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI, addvarlist_saved
