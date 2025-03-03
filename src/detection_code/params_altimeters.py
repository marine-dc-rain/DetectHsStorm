# -*- coding: utf-8 -*-
"""
Created on Sat Dec. 21 2024

@author: M. Passaro & F. ardhuin
"""

"""altimeters_parameters.py: A Python module for LRM waveform models
"""
#================================================================
# Imports
#----------------------------------------------------------------
import numpy as np
import netCDF4
from netCDF4 import Dataset
import xarray as xr

FORMAT_OUT_detalt = 'ALT_detect_storm_YYYY_MM.nc'
hs_thresh=9.
min_length=10


######################  Defines parameters
def  alti_paths_GDR(mission)  :

    PATH_ALTI_in=''
    PATH_ALTI_ii=''
    
    if mission.lower() in ['cryosat2']:
         PATH_ALTI_in = ' /home/datawork-cersat-public/provider/esa/satellite/l2/cryosat-2/siral/sir_lrm_l2/version_e/data/date/YYYY/???/CS_LTA__SIR_LRM_2__YYYYMM*.nc'
         PATH_ALTI_in = ' /home/datawork-cersat-public/provider/esa/satellite/l2/cryosat-2/siral/sir_lrm_l2/version_e/data/date/YYYY/*/*2__YYYYMM*.nc'
         TAG_ALTI='GDR_CS2'

    if mission.lower() in ['ers1']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/esa/satellite/l2/ers-1/ra/esa-reaper/ers_alt_2s/data/date/YYYY/???/E1_REAP_ERS_ALT_2S_YYYYMM*.NC'
         TAG_ALTI='GDR_ER1'

    if mission.lower() in ['ers2']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/esa/satellite/l2/ers-2/ra/esa-reaper/ers_alt_2s/data/date/YYYY/???/E2_REAP_ERS_ALT_2S_YYYYMM*.NC'
         TAG_ALTI='GDR_ER2'

    if mission.lower() in ['envisat']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/esa/satellite/l2/envisat/ra2/sgdr/v3/YYYY/???/ENV_RA_2_MWS____YYYYMM*.nc' 
         TAG_ALTI='GDR_ENV'

    if mission.lower() in ['tp-topex']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/topex-poseidon/topex/gdr/version_f/YYYY/*/TP_GPN_2PfP???_???_YYYYMM*.nc'
         TAG_ALTI='GDR_TPT'

    if mission.lower() in ['jason1']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-1/poseidon-2/sgdr/version_e/data/date/YYYY/*/JA1_GPS_2PeP???_???_YYYYMM*.nc'
         TAG_ALTI='GDR_JA1'

    if mission.lower() in ['jason2']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-2/poseidon-3/sgdr/version_d/data/date/YYYY/*/JA2_GPS_*_YYYYMM*.nc'
         TAG_ALTI='GDR_JA2'

    if mission.lower() in ['jason3f']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-3/poseidon-3b/sgdr/version_f/data/date/YYYY/*/JA3_GPS_*_YYYYMM??_*.nc'
         PATH_ALTI_ii = '/home/datawork-cersat-public/provider/aviso/satellite/l2/jason-3/poseidon-3b/igdr/version_f/data/date/YYYY/*/JA3_IPN_*_YYYYMM??_*.nc'
         TAG_ALTI='GDR_JA3'
         
    if mission.lower() in ['sentinel3a']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-3a/sral/s3a_sr_2_wat____o_st/YYYY/???/S3A_SR_2_WAT____YYYYMM*.SEN3/standard_measurement.nc'
         TAG_ALTI='GDR_S3A'

    if mission.lower() in ['sentinel3b']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-3b/sral/s3b_sr_2_wat____o_st/YYYY/???/S3B_SR_2_WAT____YYYYMM*.SEN3/standard_measurement.nc'
         TAG_ALTI='GDR_S3B'
         
    if mission.lower() in ['sentinel6']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-6a/poseidon-4/p4_2__lr/f08/YYYY/???/S6A_P4_2__LR______YYYYMM*.SEN6/S6A_P4_2__LR_STD__NT_???_???_YYYYMM*.nc'
         PATH_ALTI_ii = '/home/datawork-cersat-public/provider/eumetsat/satellite/l2/sentinel-6a/poseidon-4/p4_2__lr/f09/YYYY/???/S6A_P4_2__LR______YYYYMM*.SEN6/S6A_P4_2__LR_STD__NT_???_???_YYYYMM*.nc'
         TAG_ALTI='GDR_S6A'

    if mission.lower() in ['swot']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/swot/poseidon-3c/igdr/YYYY/???/SWOT_IPS_2PfP???_???_YYYYMM*.nc'
         PATH_ALTI_ii = '/home/datawork-cersat-public/provider/aviso/satellite/l2/swot/poseidon-3c/igdr/2.0/YYYY/???/SWOT_IPS_2PfP???_???_YYYYMM*.nc'
         TAG_ALTI='GDR_SWO'

    if mission.lower() in ['cfosat']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op??/*/YYYY/*/*F_YYYYMM*'
         TAG_ALTI='GDR_CFO'

    if mission.lower() in ['hy2a']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/aviso/satellite/l2/hy-2a/alt/rs-igdr/data/YYYY/???/H2A_RA1_RDR_2PT_????_????_YYYYMM??_*.nc'
         TAG_ALTI='GDR_H2A'

    return PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI
    
######################  Defines paths for CCI files (l2) 
def  alti_paths_cci(mission,version='1hz')  :
    print('Setting file paths for mission:',mission.lower())
    PATH_ALTI_in=''
    PATH_ALTI_ii=''
    if version=='l2p':
       rootpath='/home/datawork-cersat-public/cache/project/cciseastate/data/v4/altimeter/l2p/'
       tag1='CCI_l2p_v4_'
    else:
       rootpath='/home/datawork-cersat-public/cache/project/cciseastate/data/v4/altimeter/1hz/'
       tag1='CCIv4_'
    if mission.lower() in ['saral']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-SARAL-'
         else: 
             tag2='CS_LTA__SIR_LRM_1B_'
         PATH_ALTI_in = rootpath+'saralt/YYYY/???/'+tag2+'YYYYMM*.nc'
         PATH_ALTI_ii = rootpath+'saralf/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'SAR'

    if mission.lower() in ['cryosat2']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-CryoSat-2-'
             PATH_ALTI_in = rootpath+'cryosat-2d/YYYY/???/'+tag2+'YYYYMM*.nc'
             PATH_ALTI_ii = rootpath+'cryosat-2e/YYYY/???/'+tag2+'YYYYMM*.nc'
         else: 
             tag2='CS_LTA__SIR_LRM_1B_'
             PATH_ALTI_in = rootpath+'cryosat-2d/YYYY/???/'+tag2+'YYYYMM*.nc'
             PATH_ALTI_ii = rootpath+'cryosat-2e/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'CS2'

    if mission.lower() in ['ers1']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-ERS-1-'
         else: 
             tag2='E1_REAP_ERS_ALT_2__'
         PATH_ALTI_in = rootpath+'ers-1-reaper/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'ER1'

    if mission.lower() in ['ers2']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-ERS-2-'
         else: 
             tag2='E2_REAP_ERS_ALT_2__'
         PATH_ALTI_in = rootpath+'ers-2-reaper/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'ER2'

    if mission.lower() in ['envisat']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-Envisat-'
         else: 
             tag2='ENV_RA_2_MWS____'
         PATH_ALTI_in = rootpath+'envisat-v3/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'ENV'

    if mission.lower() in ['tp-topex']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-Topex-Poseidon-'
         else: 
             tag2='TP_GPN_2P?????_???_'
         PATH_ALTI_in = rootpath+'topexf_topex_a/YYYY/???/'+tag2+'YYYYMM*.nc'
         PATH_ALTI_ii = rootpath+'topexf_topex_b/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'TPT'

    if mission.lower() in ['jason1']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-Jason-1-'
         else: 
             tag2='JA1_GPS_2P?????_???_'
         PATH_ALTI_in = rootpath+'jason-1e/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'JA1'    
    if mission.lower() in ['jason2']:
         #PATH_ALTI_in = '/home/datawork-cersat-public/provider/cci_seastate/products/v3/data/satellite/altimeter/l2_20Hz/l2/jason-2/YYYY/????/JA2_???_2P?????_???_YYYYMM*.nc'
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-Jason-2-'
         else: 
             tag2='JA2_GPS_2P?????_???_'
         PATH_ALTI_in = rootpath+'jason-2d/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'JA2'    

    if mission.lower() in ['jason3']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-Jason-3-'
         else: 
             tag2='JA3_GPS_2P?????_???_'
         PATH_ALTI_in = rootpath+'jason-3d/YYYY/???/'+tag2+'YYYYMM*.nc'
         PATH_ALTI_ii = rootpath+'jason-3f/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'JA3'    

    if mission.lower() in ['sentinel3a']:
         PATH_ALTI_in = rootpath+'sentinel-3_a_005/YYYY/???/ESACCI-SEASTATE-L2P-SWH-Sentinel-3_A-YYYYMM*.nc'
         TAG_ALTI=tag1+'S3A'    

    if mission.lower() in ['sentinel3b']:
         PATH_ALTI_in = rootpath+'sentinel-3_b_005/YYYY/???/ESACCI-SEASTATE-L2P-SWH-Sentinel-3_B-YYYYMM*.nc'
         TAG_ALTI=tag1+'S3A'    
         TAG_ALTI=tag1+'S3B'    

    if mission.lower() in ['sentinel6a']:
         if version=='l2p':
             tag2='ESACCI-SEASTATE-L2P-SWH-Sentinel-6_A-'
         else: 
             tag2='S6A_P4_2__LR______'
         PATH_ALTI_in = rootpath+'sentinel-6_a_f08/YYYY/???/'+tag2+'YYYYMM*.nc'
         PATH_ALTI_ii = rootpath+'sentinel-6_a_f09/YYYY/???/'+tag2+'YYYYMM*.nc'
         TAG_ALTI=tag1+'S6A'    
    print('path :', PATH_ALTI_in)
    return PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI


######################  Generic reading of altimeter data: retracked 1Hz parameters
def  alti_read_l2lr(mission,filename):
    '''
    reads altimeter data (LRM 1Hz only) from file name. 
    The outout is a xarray dataset 
    '''

    S = netCDF4.Dataset(filename, 'r')
    if mission.lower() in ['ers1','ers2','ers2_r_2cm']:
    # example file='CS_LTA__SIR_LRM_2__20101231T000154_20101231T000927_E001.nc'
        swh1 = np.ma.getdata(S.variables['swh'][:])   # this is MLE4
        lat1  = np.ma.getdata(S.variables['lat'][:])
        lon1  = np.ma.getdata(S.variables['lon'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        flag1 = 3-np.ma.getdata(S.variables['swh_quality_level'][:])  

        timeref= "1981-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 
    if mission.lower() in ['cryosat2']:
    # example file='CS_LTA__SIR_LRM_2__20101231T000154_20101231T000927_E001.nc'
        swh1 = np.ma.getdata(S.variables['swh_ocean_01_ku'][:])   # this is MLE4
        lat1  = np.ma.getdata(S.variables['lat_01'][:])
        lon1  = np.ma.getdata(S.variables['lon_01'][:])
        time1 = np.ma.getdata(S.variables['time_cor_01'][:])
        flag1 = swh1*0. #np.ma.getdata(S.variables['retracker_1_quality_20_ku'][:])
        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission.lower() in ['jason1','jason2','jason3']:
    # example file='JA2_GPS_2PdP011_200_20081026_233206_20081027_002819.nc'
        swh1 = np.ma.getdata(S.variables['swh_ku'][:])   # this is MLE4
        lat1  = np.ma.getdata(S.variables['lat'][:])
        lon1  = np.ma.getdata(S.variables['lon'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        flag1 = np.ma.getdata(S.variables['qual_alt_1hz_swh_ku'][:])
        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission.lower() in ['jason3f','swot']:
    # example file='JA2_GPS_2PdP011_200_20081026_233206_20081027_002819.nc'
        swh1 = np.ma.getdata(S['data_01']['ku'].variables['swh_ocean'][:])   # this is MLE4
        lat1  = np.ma.getdata(S['data_01'].variables['latitude'][:])
        lon1  = np.ma.getdata(S['data_01'].variables['longitude'][:])
        time1 = np.ma.getdata(S['data_01'].variables['time'][:])
        flag1 = np.ma.getdata(S['data_01']['ku'].variables['swh_ocean_compression_qual'][:])
        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission in ['saral', 'altika']:
    # example file='SRL_GPS_2PfP001_0641_20130405_141055_20130405_150113.CNES.nc'
        swh1 = np.ma.getdata(S.variables['swh'][:])   # this is MLE4?
        lat1  = np.ma.getdata(S.variables['lat'][:])
        lon1  = np.ma.getdata(S.variables['lon'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        flag1 = np.ma.getdata(S.variables['qual_alt_1hz_swh'][:])
        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission.lower() in ['cfosat']:
    # example file='CFO_OP07_SWI_L2_____F_20241221T043021_20241221T054444.nc'
        S = netCDF4.Dataset(filename, 'r')
        swh1 = np.ma.getdata(S.variables['nadir_swh_1Hz'][:])
    #    swh_sgdr2 = np.ma.getdata(S.variables['nadir_swh_native'][:])
    #S_waveform = np.ma.getdata(S.variables['waveforms_40hz'][:])
        lat1  = np.ma.getdata(S.variables['lat_nadir_1Hz'][:])
        lon1  = np.ma.getdata(S.variables['lon_nadir_1Hz'][:])
#        time1 = np.ma.getdata(S.variables['time_nadir_1Hz'][:])
        time1 = S.variables['time_nadir_1Hz'][:]
        flag1 = np.ma.getdata(S.variables['flag_valid_swh_1Hz'][:])
        timeref= "2009-01-01 00:00:00 0:00"
    #reference_time = pd.Timestamp("2014-09-05")
    ds = xr.Dataset(
        {   "swh_1hz": (["time"], swh1),
            "lon_1hz": (["time"], lon1),
            "lat_1hz": (["time"], lat1),
            "flag_1hz": (["time"], flag1),
        },
        coords={
            "time": time1,
            "reference_time": timeref,
        },
        )
    return ds
    
######################  Generic reading of altimeter data: retracked 1Hz parameters
def  alti_read_l2lr_cci(mission,filename,version=''):
    '''
    reads altimeter data (LRM 1Hz only) from file name. 
    The outout is a xarray dataset 
    NB: here are te variables in ccp l2p V4 : dimensions(sizes): time(25)
    variables(dimensions): float64 time(time), float64 lon(time), float64 lat(time), float64 swh(time), float64 swh_rms(time), int8 swh_numval(time), int8 swh_quality_level(time), int8 swh_rejection_flags(time), float64 swh_corrected(time), float64 swh_corrected_rms(time), int8 swh_corrected_numval(time), int8 swh_corrected_quality_level(time), int8 swh_corrected_rejection_flags(time), float64 sigma0_ku(time), float64 sigma0_ku_rms(time), int8 sigma0_ku_numval(time), int8 sigma0_ku_quality_level(time), int8 sigma0_ku_rejection_flags(time), int32 distance_to_coast(time), int32 bathymetry(time), float64 sea_ice_fraction(time), float32 era5_tclw(time), float32 era5_t2m(time), float32 era5_sst(time), float32 era5_u10(time), float32 era5_v10(time), float32 era5_sp(time), float32 era5wave_p140121(time), float32 era5wave_mpww(time), float32 era5wave_swh(time), float32 era5wave_p1ps(time), float32 era5wave_pp1d(time), float32 era5wave_shww(time), float32 era5wave_mwp(time), float32 era5wave_p140122(time), float32 era5wave_mwd(time), float32 era5wave_mdww(time), float32 ww3_uwnd(time), float32 ww3_t01(time), float32 ww3_fp(time), float32 ww3_vwnd(time), float32 ww3_t02(time), float32 ww3_hs(time), float32 ww3_dir(time)
    '''

    import xarray as xr
    try:
        # Attempt to open the NetCDF file
        S = netCDF4.Dataset(filename, 'r')
        readOK=1
    except OSError as e:
        # Handle OSError (such as NetCDF or HDF errors)
        #logging.error(f"################# Failed to open {file_path}: {str(e)}")
        print(f"################# Failed to open {filename}: {str(e)}")
        readOK=0
        return readOK,[]

    if ('swh' in S.variables) and ('lon' in S.variables):
        try:
            swh1 = np.ma.getdata(S.variables['swh'][:])   
            rms1  = np.ma.getdata(S.variables['swh_rms'][:])
            lat1  = np.ma.getdata(S.variables['lat'][:])
            lon1  = np.ma.getdata(S.variables['lon'][:])
            time1 = np.ma.getdata(S.variables['time'][:])
            flag1 = 3-np.ma.getdata(S.variables['swh_quality_level'][:])
            timeref= "1981-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 
        except (OSError, RuntimeError) as e:
            readOK=0
            return readOK,[]
    else: 
        readOK=0
        print('################# ',filename,' only has these vars:',S.variables)
        return readOK,[]

    ds = xr.Dataset(
        {   "swh_1hz": (["time"], swh1),
            "swh_rms": (["time"], rms1),
            "lon_1hz": (["time"], lon1),
            "lat_1hz": (["time"], lat1),
            "flag_1hz": (["time"], flag1),
        },
        coords={
            "time": time1,
            "reference_time": timeref,
        },
        )

    if version=='l2p':
       if 'swh_corrected_denoised' in S.variables:
          denoisevar='swh_corrected_denoised'
          denoiseunc='swh_corrected_denoised_uncertainty'
       else:
          denoisevar='swh_denoised'
          denoiseunc='swh_denoised_uncertainty'
       ds = ds.rename({'swh_1hz': denoisevar})
       l2paddvars=['ww3_hs','ww3_qkk',denoisevar,denoiseunc,'era5wave_swh']
       for addvar in l2paddvars: 
           if (addvar in S.variables):
               try:
                   thisvar=np.ma.getdata(S.variables[addvar][:])
                   ds=ds.assign({addvar : (('time'),thisvar)})
               except (OSError, RuntimeError) as e:
                   readOK=0
                   return readOK,[]

           else: 
               readOK=0
               print((addvar in S.variables),'################# Failed to read ',addvar,'in ',filename)
               return readOK,ds

       ds = ds.rename({denoisevar: 'swh_1hz'})
       ds = ds.rename({denoiseunc: 'swh_denoised_uncertainty'})
       ds = ds.assign({'swh_noisy' : (('time'),swh1)})
    return readOK,ds

######################  Generic reading of altimeter data: waveforms and 20Hz parameters
def  alti_read_l2hrw(mission,filename):
    '''
    reads altimeter data (LRM 1Hz only) from file name. 
    The outout is a xarray dataset 
    '''

    S = netCDF4.Dataset(filename, 'r')
    if mission.lower() in ['cryosat2']:
        #Time at 1-Hz for interpolation of fields available only at 1-Hz
        time1=np.ma.getdata(S.variables['time_cor_01'][:])
        S_time = np.ma.getdata(S.variables['time_20_ku'][:])

        #S_height = np.ma.getdata(S.variables['alt_20_ku'][:])
        #S_height = np.reshape(S_height, (np.shape(S_time)[0], 1))

        
      #  tracker1= np.ma.getdata( S.variables['window_del_20_ku'][:] * (299792458.0) / 2.0)      # warning : light speed should be constant 
      #  range1=S_range=np.ma.getdata( S.variables['range_ocean_20_ku'][:] )
        waveforms= np.ma.getdata(S.variables['pwr_waveform_20_ku'][:]).astype(np.float64)
      # S_waveform=np.reshape(S_waveform,(np.shape(S_time)[0],1) )

        S_lat=np.ma.getdata(S.variables['lat_20_ku'][:])
        S_lon=  np.ma.getdata(S.variables['lon_20_ku'][:])

        #off=S_offnadir=np.ma.getdata( S.variables[' off_nadir_roll_angle_str_20_ku'][:] )     #degrees ... must add pitch ... 
        #flag1=np.ma.getdata(S.variables['retracker_1_quality_20_ku'][:])
       
        nhf=20
        n1=len(time1)
        nall=len(S_time)
        nlr=nall//nhf
        nal=nlr*nhf

        S_time = np.reshape(S_time[0:nal], (nlr,nhf))   # warning this reshaping throws away the last few 20 Hz sample 
        #print('time shape:',n1,S_time.shape)
       
        lat1 = np.reshape(S_lat[0:nal], (nlr,nhf))
        lon1 =np.reshape(S_lon[0:nal], (nlr,nhf))
        flag1=lon1*0+np.nan
        
        swh1 = lon1*0+np.nan
       
        S_waveform =  np.ma.getdata(S.variables['pwr_waveform_20_ku'][:]).astype(np.float64)
        waveforms = np.reshape(
        S_waveform[0:nal,:], (nlr, nhf, np.shape(S_waveform)[1]))

        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission.lower() in ['ers1','ers2']:
        #Time at 1-Hz for interpolation of fields available only at 1-Hz
        time1=np.ma.getdata( S.variables['time'][:] )
        S_time = np.ma.getdata(S.variables['time_20hz'][:]).flatten()
        nhf=20
        n1=len(time1)
        nall=len(S_time)
        nlr=nall//nhf
        nal=nlr*nhf

        S_time=np.ma.getdata( S.variables['time_20hz'][:] )
        #S_time=np.reshape(S_time,(np.shape(S_time)[0],1) )
    
    
        height1=np.ma.getdata( S.variables['alt_20hz'][:] )
        swh1=np.ma.getdata( S.variables['swh_20hz'][:] )
        tracker1=np.ma.getdata( S.variables['tracker_range_20hz'][:] )
        range1=np.ma.getdata( S.variables['ocean_range_20hz'][:] )
        waveforms=np.ma.getdata( S.variables['ku_wf'][:] ).astype(np.float64)

        lat1=np.ma.getdata( S.variables['lat_20hz'][:] )
        lon1=np.ma.getdata( S.variables['lon_20hz'][:] )
        off=np.ma.getdata( S.variables['off_nadir_angle_wf_20hz'][:] )
        off=np.zeros_like(off) # THE OFF NADIR FIELD IN ERS2 IS EMPTY!!!!!

        atmos_corr=np.ma.getdata( S.variables['atmos_corr_sig0'][:] )
        atmos_corr=np.transpose(np.tile(atmos_corr,(np.shape(S_time)[1],1)))
        scaling_factor=np.ma.getdata( S.variables['scaling_factor_20hz'][:] )
        flag1 = np.ma.getdata(S.variables['ocean_mqe_20hz'][:]) 
        timeref= "1981-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

        
    if mission.lower() in ['jason1','jason2','jason3']:
    # example file='JA2_GPS_2PdP011_200_20081026_233206_20081027_002819.nc'
        time1 = np.ma.getdata(S.variables['time'][:])
        S_time = np.ma.getdata(S.variables['time_20hz'][:]).flatten()
        nhf=20
        n1=len(time1)
        nall=len(S_time)
        nlr=nall//nhf
        nal=nlr*nhf
        swh1 = np.ma.getdata(S.variables['swh_20hz_ku'][:])  # this is MLE4
        lat1= np.ma.getdata(S.variables['lat_20hz'][:])
        lon1= np.ma.getdata(S.variables['lon_20hz'][:])

                
        waveforms = np.ma.getdata(S.variables['waveforms_20hz_ku'][:])

        flag1 = np.ma.getdata(S.variables['qual_alt_1hz_swh_ku'][:])
        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission.lower() in ['jason3f','swot']:
        # AVISO SGDR version F
        time1 = np.ma.getdata(S['data_01'].variables['time'][:])
        S_time = np.ma.getdata(S['data_20'].variables['time'][:])
        nhf=20
        n1=len(time1)
        nall=len(S_time)
        nlr=nall//nhf
        nal=nlr*nhf
        
        #print('time:',time1[0],'##',S_time[0:20]-time1[0])
        #print('time:',time1[nlr-1],'##',S_time[(nlr-1)*nhf:nlr*nhf]-time1[nlr-1])
        S_time = np.reshape(S_time[0:nal], (nlr,nhf))   # warning this reshaping throws away the last few 20 Hz sample 
        #print('time shape:',n1,S_time.shape)
       
        S_swh = np.ma.getdata(S['data_20']['ku'].variables['swh_ocean'][:]) # this is MLE4
        swh1 = np.reshape(S_swh[0:nal], (nlr,nhf))

        S_lat= np.ma.getdata(S['data_20'].variables['latitude'][:])
        lat1 = np.reshape(S_lat[0:nal], (nlr,nhf))

        S_lon= np.ma.getdata(S['data_20'].variables['longitude'][:])
        lon1 =np.reshape(S_lon[0:nal], (nlr,nhf))
        
        S_waveform = np.ma.getdata(
        S['data_20']['ku'].variables['power_waveform'][:])
        waveforms = np.reshape(
        S_waveform[0:nal,:], (nlr, nhf, np.shape(S_waveform)[1]))
        
        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission in ['saral', 'altika']:
    # example file='SRL_GPS_2PfP001_0641_20130405_141055_20130405_150113.CNES.nc'
        swh2 = np.ma.getdata(S.variables['swh_40hz'][:])   # this is MLE4?
        lat2  = np.ma.getdata(S.variables['lat_40hz'][:])
        lon2  = np.ma.getdata(S.variables['lon_40hz'][:])
        time2 = np.ma.getdata(S.variables['time_40hz'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        off2 = np.ma.getdata(S.variables['off_nadir_angle_pf'][:])
        waveform = np.ma.getdata(S.variables['waveforms_40hz'][:])

        timeref= "2000-01-01 00:00:00.0"			# WARNING: this should be read from the attribute of the time variable ... 

    if mission.lower() in ['cfosat']:
    # example file='CFO_OP07_SWI_L2_____F_20241221T043021_20241221T054444.nc'
        S = netCDF4.Dataset(filename, 'r')
        swh1 = np.ma.getdata(S.variables['nadir_swh_1Hz'][:])
    #    swh_sgdr2 = np.ma.getdata(S.variables['nadir_swh_native'][:])
        waveforms = np.ma.getdata(S.variables['waveforms_40hz'][:])
        lat1  = np.ma.getdata(S.variables['lat_nadir_1Hz'][:])
        lon1  = np.ma.getdata(S.variables['lon_nadir_1Hz'][:])
#        time1 = np.ma.getdata(S.variables['time_nadir_1Hz'][:])
        time1 = S.variables['time_nadir_1Hz'][:]
        flag1 = np.ma.getdata(S.variables['flag_valid_swh_1Hz'][:])
        timeref= "2009-01-01 00:00:00 0:00"
    #reference_time = pd.Timestamp("2014-09-05")
    ds = xr.Dataset(
        {   "swh2d": (["time","meas_ind"], swh1),
            "lon2d": (["time","meas_ind"], lon1),
            "lat2d": (["time","meas_ind"], lat1),
            "flag2d":(["time","meas_ind"], flag1),
            "waveforms": (["time","meas_ind","wvf_ind"], waveforms),
        },
        coords={
            "time": time1[0:nlr],
            "reference_time": timeref,
        },
        )
    return ds

##########################################################
def get_storm_info_from_savemap(ds):
    '''
    function to be applied by a "MAP" function of xarray: here I tried to do the same as was done for model ... 
    '''
    idm = ds.swh_1hz.argmax()
    print('IDM:',idm)
    lensum = (ds.swh_1hz*0+1).sum()
    hsmax = ds.swh_1hz.max()

    dsn = xr.Dataset({'lon_max' : (["x"], (ds.lon_1hz[idm].values+180)%360 -180), 
                      'lat_max' : (["x"], ds.lat_1hz[idm].values), 
                      'time' :    (["x"], ds.time[idm].values), 
                      'hs_max' :  (["x"], hsmax),
                      'sizestorm':(["x"], lensum), 
                     }, 
                     coords={
                      "x": ds.segments[idm].values,
                     }, 
                    )
    print('DSN:',dsn)
    return dsn

##########################################################
# def get_storm_by_timestep(ds1,levels,Npix_min,amp_thresh, d_thresh_min, d_thresh_max, min_area,  area_forgotten_ratio, plot_output = False, plot_example = False):
def get_storm_by_file(mission,origin,filename,yy,mm,hs_thresh,min_len, count0=0,plot_output = False, plot_example = False):   
    # --- concat [-180: 360] ----

    res=[]
    ds3=[]
    count1=count0
    if origin=='gdr':
        ds=alti_read_l2lr(mission,filename)
    else:
        readOK,ds=alti_read_l2lr_cci(mission,filename,version=origin)
    if (readOK==0): 
        print('Problem in reading from '+origin+' file:',filename)  
        return ds3,res,count1
   
    #inds=np.where(flag1 > 0)[0]
#   First step: finds points above threshold 
    inds=np.where((ds.swh_1hz > hs_thresh) & (ds.flag_1hz ==0 ))[0]
    indb=np.where((ds.swh_1hz > 1000.) | (ds.flag_1hz > 0 ))[0]
    ds.swh_1hz[indb]=0.
    if len(inds > min_len):
       #print('count0:',count0,'INDS:',inds) 
       to_save = np.zeros_like(ds['swh_1hz'].data,dtype='int')-100
       inddif=np.diff(inds)
       
       # define indices of segment boundaries 
       endseg=np.append(np.where(inddif > min_len)[0],[len(inds)-1])

       i1=0
       i2=len(inds)-1
       countst=0
       
       for iseg in range(len(endseg)):
           i2=endseg[iseg]
           if (inds[i2] > inds[i1]+min_len): 
               if (np.sum(inddif[i1:i2-1]) < 2*(i2-i1)):
                   to_save[inds[i1]:inds[i2]] = countst+count0
                   countst = countst + 1
           i1=i2+1
       
       if countst > 0:
           ds3=ds.assign({'segments' :(('time'),to_save)})
           ind3=np.where((ds3.segments > -1 ))[0]
           print('START:',inds[0],inds[-1],ds.swh_1hz[inds[0]].values,np.max(ds.swh_1hz[inds[:]].values))
           res=[]
           g=ds3.where(ds3.segments>-1.,np.nan).groupby('segments')
           #print('GGG:',g)
           #res = g.map(get_storm_info_from_savemap) #.swap_dims({'time':'x'})
           ds3 = ds3.isel(time=inds).swap_dims({'time':'x'})
           ninds=len(inds)
           iarr=np.zeros(ninds,dtype='uint')+yy*100+mm
           ds3=ds3.assign({'yearmonth' :(('x'),iarr)})
           sarr = np.array([filename for _ in range(ninds)], dtype=str)
           ds3=ds3.assign({'filename' :(('x'),sarr)})

           ds3=ds3.assign({'indices_in_file' :(('x'),inds)})
           
           #ds3 = ds3.assign_coords(storms_by_t = xr.full_like(ds3.segments,fill_value=len(g),dtype=int))
           #print('RES:',res)
           #res = res.assign_coords(storms_by_t = xr.full_like(res.segments,fill_value=len(g),dtype=int))
           #print('RES2:',res)
           #print('DS3:',ds3)
       count1=count0+countst
      
    if plot_example:
        plt.figure(fig)
        plt.tight_layout()  
        fig_end = plt.figure(figsize=(12,5))
#        im = plt.pcolormesh(ds3.longitude,ds3.latitude,to_save_intersec,cmap='jet',vmin=-1)
#        plt.colorbar(im)
#        plt.plot(res.lon_max,res.lat_max,'ow',markeredgecolor='k')
#        plt.title('detected regions')
    if plot_output:
        fig_end2 = plt.figure(figsize=(12,5))
#        im = plt.pcolormesh(ds3.longitude,ds3.latitude,dsTot.hs,cmap='jet',vmin=-1)

#        plt.contour(ds3.longitude,ds3.latitude,(to_save_intersec>-1)*1,levels=[0.5],colors='m',linewidths=2)
#        plt.contour(ds3.longitude,ds3.latitude,(to_save_intersec>-1)*1,levels=[0.5],colors='w',linewidths=1)
#        plt.colorbar(im)
#        plt.plot(res.lon_max,res.lat_max,'ow',markeredgecolor='k')
#        plt.title('Hs with detected regions')
        return res, fig_end2
    else:
        return ds3,res,count1


