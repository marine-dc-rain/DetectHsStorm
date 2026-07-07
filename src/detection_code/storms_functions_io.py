import os
import pathlib
import sys

import numpy as np
import xarray as xr
import netCDF4
from netCDF4 import Dataset

from detection_code.storms_functions_geo import lat_lon_cell_area

'''
Functions :
- py_files(root,suffix='.nc')
I/O for ERA5
- preprocessing_ERA5(ds0)
- read_ERA5_HS_file(Files) : opens multiple ERA5 daily files (preprocessing_ERA5 gives information on their concatenation)
- read_ERA5_HS_1file(filename) : opens 1 ERA5 daily files (preprocessing_ERA5 gives information on their concatenation)
- read_ERA5c_HS_file(filename) : 2nd reader to open 1 ERA5 file (preprocessing_ERA5 gives information on their concatenation)
I/O for WW3
- read_WW3_HS_file(PATH,filename) : open 1 monthly WW3 file + rename variable as needed
I/O for sat
- alti_read_l2lr(mission, filename) : Generic reading of altimeter data: retracked 1Hz parameters
- alti_read_l2lr_cci(mission, filename, version='') : Generic reading of altimeter data: retracked 1Hz parameters (CCI v4)
- alti_read_l2hrw(mission, filename) : Generic reading of altimeter data: waveforms and 20Hz parameters
'''


def py_files(root, suffix='.nc'):
    """Recursively iterate all the .nc files in the root directory and below"""
    for path, dirs, files in os.walk(root):
        if suffix[0] == '.':
            yield from (os.path.join(path, file) for file in files if pathlib.Path(file).suffix == suffix)
        else:
            yield from (os.path.join(path, file) for file in files if file[-len(suffix) :] == suffix)


#### --- I/O for ERA 5 ---------------------
def preprocessing_ERA5(ds0):
    ds = ds0[['longitude050', 'latitude050', 'time', 'swh', 'mwd', 'pp1d', 'mwp', 'u10', 'v10']]
    all_lats, all_lons = np.meshgrid(ds0['latitude050'].data, ds0['longitude050'].data, indexing='ij')
    side_length = 0.5

    lat_lon_grid_cell = np.array(
        [all_lons, all_lats - side_length / 2, all_lons + side_length, all_lats + side_length / 2]
    )
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6

    ds = ds.rename_dims({'longitude050': 'longitude'}).rename_vars({'longitude050': 'longitude'})
    ds = ds.rename_dims({'latitude050': 'latitude'}).rename_vars({'latitude050': 'latitude'})
    ds = ds.rename_vars({'swh': 'hs'})
    ds = ds.rename_vars({'u10': 'uwnd'})
    ds = ds.rename_vars({'v10': 'vwnd'})
    ds = ds.rename_vars({'pp1d': 'fp'})
    ds = ds.rename_vars({'mwd': 'dir'})
    ds = ds.rename_vars({'mwp': 'f0m1'})

    return ds.assign({'areakm2': (('latitude', 'longitude'), areakm2)})


def read_ERA5_HS_file(Files):
    DS = xr.open_mfdataset(
        Files, preprocess=preprocessing_ERA5, concat_dim='time', combine='nested'
    )  # ,dask='forbidden') #chunks={'time': 10})
    return DS.compute()


def read_ERA5_HS_1file(filename):
    ds0 = xr.open_dataset(filename)
    ds = ds0[['longitude050', 'latitude050', 'time', 'swh', 'mwd', 'pp1d', 'mwp']]
    all_lats, all_lons = np.meshgrid(ds0['latitude050'].data, ds0['longitude050'].data, indexing='ij')
    side_length = 0.5

    lat_lon_grid_cell = np.array(
        [all_lons, all_lats - side_length / 2, all_lons + side_length, all_lats + side_length / 2]
    )
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6

    ds = ds.rename_dims({'longitude050': 'longitude'}).rename_vars({'longitude050': 'longitude'})
    ds = ds.rename_dims({'latitude050': 'latitude'}).rename_vars({'latitude050': 'latitude'})
    ds = ds.rename_vars({'swh': 'hs'})
    ds = ds.rename_vars({'pp1d': 'fp'})
    ds = ds.rename_vars({'mwd': 'dir'})
    ds = ds.rename_vars({'mwp': 't0m1'})

    return ds.assign({'areakm2': (('latitude', 'longitude'), areakm2)})


def read_ERA5c_HS_file(filename):
    ds0 = xr.open_dataset(filename)
    ds = ds0[['longitude', 'latitude', 'time', 'hs']]
    all_lats, all_lons = np.meshgrid(ds0['latitude'].data, ds0['longitude'].data, indexing='ij')
    side_length = 0.5

    lat_lon_grid_cell = np.array(
        [all_lons, all_lats - side_length / 2, all_lons + side_length, all_lats + side_length / 2]
    )
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6

    return ds.assign({'areakm2': (('latitude', 'longitude'), areakm2)})


#### --- I/O for WW3 ---------------------
def read_WW3_HS_file(filename):
    print('reading file:', filename)
    ds0 = xr.open_dataset(filename)
    #    ds = ds0[['longitude','latitude','time','hs']]
    ds = ds0[['longitude', 'latitude', 'time', 'hs', 'dir', 't0m1', 'fp', 'spr', 'uwnd', 'vwnd']]
    all_lats, all_lons = np.meshgrid(ds0['latitude'].data, ds0['longitude'].data, indexing='ij')
    side_length = 0.5

    lat_lon_grid_cell = np.array(
        [all_lons, all_lats - side_length / 2, all_lons + side_length, all_lats + side_length / 2]
    )
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6

    return ds.assign({'areakm2': (('latitude', 'longitude'), areakm2)})


#### --- I/O for Sats ---------------------
######################  Generic reading of altimeter data: retracked 1Hz parameters
def alti_read_l2lr(mission, filename):
    '''
    reads altimeter data (LRM 1Hz only) from file name.
    The outout is a xarray dataset
    '''

    S = netCDF4.Dataset(filename, 'r')
    if mission.lower() in ['ers1', 'ers2', 'ers2_r_2cm']:
        # example file='CS_LTA__SIR_LRM_2__20101231T000154_20101231T000927_E001.nc'
        swh1 = np.ma.getdata(S.variables['swh'][:])  # this is MLE4
        lat1 = np.ma.getdata(S.variables['lat'][:])
        lon1 = np.ma.getdata(S.variables['lon'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        flag1 = 3 - np.ma.getdata(S.variables['swh_quality_level'][:])

        timeref = "1981-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...
    if mission.lower() in ['cryosat2']:
        # example file='CS_LTA__SIR_LRM_2__20101231T000154_20101231T000927_E001.nc'
        swh1 = np.ma.getdata(S.variables['swh_ocean_01_ku'][:])  # this is MLE4
        lat1 = np.ma.getdata(S.variables['lat_01'][:])
        lon1 = np.ma.getdata(S.variables['lon_01'][:])
        time1 = np.ma.getdata(S.variables['time_cor_01'][:])
        flag1 = swh1 * 0.0  # np.ma.getdata(S.variables['retracker_1_quality_20_ku'][:])
        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['jason1', 'jason2', 'jason3']:
        # example file='JA2_GPS_2PdP011_200_20081026_233206_20081027_002819.nc'
        swh1 = np.ma.getdata(S.variables['swh_ku'][:])  # this is MLE4
        lat1 = np.ma.getdata(S.variables['lat'][:])
        lon1 = np.ma.getdata(S.variables['lon'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        flag1 = np.ma.getdata(S.variables['qual_alt_1hz_swh_ku'][:])
        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['jason3f', 'swot']:
        # example file='JA2_GPS_2PdP011_200_20081026_233206_20081027_002819.nc'
        swh1 = np.ma.getdata(S['data_01']['ku'].variables['swh_ocean'][:])  # this is MLE4
        lat1 = np.ma.getdata(S['data_01'].variables['latitude'][:])
        lon1 = np.ma.getdata(S['data_01'].variables['longitude'][:])
        time1 = np.ma.getdata(S['data_01'].variables['time'][:])
        flag1 = np.ma.getdata(S['data_01']['ku'].variables['swh_ocean_compression_qual'][:])
        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission in ['saral', 'altika']:
        # example file='SRL_GPS_2PfP001_0641_20130405_141055_20130405_150113.CNES.nc'
        swh1 = np.ma.getdata(S.variables['swh'][:])  # this is MLE4?
        lat1 = np.ma.getdata(S.variables['lat'][:])
        lon1 = np.ma.getdata(S.variables['lon'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        flag1 = np.ma.getdata(S.variables['qual_alt_1hz_swh'][:])
        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['cfosat']:
        # example file='CFO_OP07_SWI_L2_____F_20241221T043021_20241221T054444.nc'
        S = netCDF4.Dataset(filename, 'r')
        swh1 = np.ma.getdata(S.variables['nadir_swh_1Hz'][:])
        #    swh_sgdr2 = np.ma.getdata(S.variables['nadir_swh_native'][:])
        # S_waveform = np.ma.getdata(S.variables['waveforms_40hz'][:])
        lat1 = np.ma.getdata(S.variables['lat_nadir_1Hz'][:])
        lon1 = np.ma.getdata(S.variables['lon_nadir_1Hz'][:])
        #        time1 = np.ma.getdata(S.variables['time_nadir_1Hz'][:])
        time1 = S.variables['time_nadir_1Hz'][:]
        flag1 = np.ma.getdata(S.variables['flag_valid_swh_1Hz'][:])
        timeref = "2009-01-01 00:00:00 0:00"
    # reference_time = pd.Timestamp("2014-09-05")
    ds = xr.Dataset(
        {
            "swh_1hz": (["time"], swh1),
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
def alti_read_l2lr_cci(mission, filename, version=''):
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
        readOK = 1
    except OSError as e:
        # Handle OSError (such as NetCDF or HDF errors)
        # logging.error(f"################# Failed to open {file_path}: {str(e)}")
        print(f"################# Failed to open {filename}: {str(e)}")
        readOK = 0
        return readOK, []

    if ('swh' in S.variables) and ('lon' in S.variables):
        try:
            swh1 = np.ma.getdata(S.variables['swh'][:])
            rms1 = np.ma.getdata(S.variables['swh_rms'][:])
            lat1 = np.ma.getdata(S.variables['lat'][:])
            lon1 = np.ma.getdata(S.variables['lon'][:])
            time1 = np.ma.getdata(S.variables['time'][:])
            flag1 = 3 - np.ma.getdata(S.variables['swh_quality_level'][:])
            timeref = (
                "1981-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...
            )
        except (OSError, RuntimeError) as e:
            readOK = 0
            return readOK, []
    else:
        readOK = 0
        print('################# ', filename, ' only has these vars:', S.variables)
        return readOK, []

    ds = xr.Dataset(
        {
            "swh_1hz": (["time"], swh1),
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

    if version == 'l2p':
        if 'swh_corrected_denoised' in S.variables:
            denoisevar = 'swh_corrected_denoised'
            denoiseunc = 'swh_corrected_denoised_uncertainty'
        else:
            denoisevar = 'swh_denoised'
            denoiseunc = 'swh_uncertainty'
        ds = ds.rename({'swh_1hz': denoisevar})
        l2paddvars = ['ww3_hs', 'ww3_qkk', denoisevar, denoiseunc, 'era5wave_swh']
        l2paddvars = [
            'ww3_swh',
            'ww3_wavenumber_peakdness',
            denoisevar,
            denoiseunc,
            'era5_swh',
            'swh_rms',
            'swh_numval',
        ]
        l2paddvars = list(set(l2paddvars + addvarlist))
        for addvar in l2paddvars:
            if addvar in S.variables:
                try:
                    thisvar = np.ma.getdata(S.variables[addvar][:])
                    ds = ds.assign({addvar: (('time'), thisvar)})
                except (OSError, RuntimeError) as e:
                    readOK = 0
                    return readOK, []

            else:
                readOK = 0
                print((addvar in S.variables), '################# Failed to read ', addvar, 'in ', filename)
                return readOK, ds

        ds = ds.rename({denoisevar: 'swh_1hz'})
        ds = ds.rename({denoiseunc: 'swh_denoised_uncertainty'})
        ds = ds.assign({'swh_noisy': (('time'), swh1)})
    return readOK, ds


######################  Generic reading of altimeter data: waveforms and 20Hz parameters
def alti_read_l2hrw(mission, filename):
    '''
    reads altimeter data (LRM 1Hz only) from file name.
    The outout is a xarray dataset
    '''

    S = netCDF4.Dataset(filename, 'r')
    if mission.lower() in ['cryosat2']:
        # Time at 1-Hz for interpolation of fields available only at 1-Hz
        time1 = np.ma.getdata(S.variables['time_cor_01'][:])
        S_time = np.ma.getdata(S.variables['time_20_ku'][:])

        # S_height = np.ma.getdata(S.variables['alt_20_ku'][:])
        # S_height = np.reshape(S_height, (np.shape(S_time)[0], 1))

        #  tracker1= np.ma.getdata( S.variables['window_del_20_ku'][:] * (299792458.0) / 2.0)      # warning : light speed should be constant
        #  range1=S_range=np.ma.getdata( S.variables['range_ocean_20_ku'][:] )
        waveforms = np.ma.getdata(S.variables['pwr_waveform_20_ku'][:]).astype(np.float64)
        # S_waveform=np.reshape(S_waveform,(np.shape(S_time)[0],1) )

        S_lat = np.ma.getdata(S.variables['lat_20_ku'][:])
        S_lon = np.ma.getdata(S.variables['lon_20_ku'][:])

        # off=S_offnadir=np.ma.getdata( S.variables[' off_nadir_roll_angle_str_20_ku'][:] )     #degrees ... must add pitch ...
        # flag1=np.ma.getdata(S.variables['retracker_1_quality_20_ku'][:])

        nhf = 20
        n1 = len(time1)
        nall = len(S_time)
        nlr = nall // nhf
        nal = nlr * nhf

        S_time = np.reshape(S_time[0:nal], (nlr, nhf))  # warning this reshaping throws away the last few 20 Hz sample
        # print('time shape:',n1,S_time.shape)

        lat1 = np.reshape(S_lat[0:nal], (nlr, nhf))
        lon1 = np.reshape(S_lon[0:nal], (nlr, nhf))
        flag1 = lon1 * 0 + np.nan

        swh1 = lon1 * 0 + np.nan

        S_waveform = np.ma.getdata(S.variables['pwr_waveform_20_ku'][:]).astype(np.float64)
        waveforms = np.reshape(S_waveform[0:nal, :], (nlr, nhf, np.shape(S_waveform)[1]))

        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['ers1', 'ers2']:
        # Time at 1-Hz for interpolation of fields available only at 1-Hz
        time1 = np.ma.getdata(S.variables['time'][:])
        S_time = np.ma.getdata(S.variables['time_20hz'][:]).flatten()
        nhf = 20
        n1 = len(time1)
        nall = len(S_time)
        nlr = nall // nhf
        nal = nlr * nhf

        S_time = np.ma.getdata(S.variables['time_20hz'][:])
        # S_time=np.reshape(S_time,(np.shape(S_time)[0],1) )

        height1 = np.ma.getdata(S.variables['alt_20hz'][:])
        swh1 = np.ma.getdata(S.variables['swh_20hz'][:])
        tracker1 = np.ma.getdata(S.variables['tracker_range_20hz'][:])
        range1 = np.ma.getdata(S.variables['ocean_range_20hz'][:])
        waveforms = np.ma.getdata(S.variables['ku_wf'][:]).astype(np.float64)

        lat1 = np.ma.getdata(S.variables['lat_20hz'][:])
        lon1 = np.ma.getdata(S.variables['lon_20hz'][:])
        off = np.ma.getdata(S.variables['off_nadir_angle_wf_20hz'][:])
        off = np.zeros_like(off)  # THE OFF NADIR FIELD IN ERS2 IS EMPTY!!!!!

        atmos_corr = np.ma.getdata(S.variables['atmos_corr_sig0'][:])
        atmos_corr = np.transpose(np.tile(atmos_corr, (np.shape(S_time)[1], 1)))
        scaling_factor = np.ma.getdata(S.variables['scaling_factor_20hz'][:])
        flag1 = np.ma.getdata(S.variables['ocean_mqe_20hz'][:])
        timeref = "1981-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['jason1', 'jason2', 'jason3']:
        # example file='JA2_GPS_2PdP011_200_20081026_233206_20081027_002819.nc'
        time1 = np.ma.getdata(S.variables['time'][:])
        S_time = np.ma.getdata(S.variables['time_20hz'][:]).flatten()
        nhf = 20
        n1 = len(time1)
        nall = len(S_time)
        nlr = nall // nhf
        nal = nlr * nhf
        swh1 = np.ma.getdata(S.variables['swh_20hz_ku'][:])  # this is MLE4
        lat1 = np.ma.getdata(S.variables['lat_20hz'][:])
        lon1 = np.ma.getdata(S.variables['lon_20hz'][:])

        waveforms = np.ma.getdata(S.variables['waveforms_20hz_ku'][:])

        flag1 = np.ma.getdata(S.variables['qual_alt_1hz_swh_ku'][:])
        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['jason3f', 'swot']:
        # AVISO SGDR version F
        time1 = np.ma.getdata(S['data_01'].variables['time'][:])
        S_time = np.ma.getdata(S['data_20'].variables['time'][:])
        nhf = 20
        n1 = len(time1)
        nall = len(S_time)
        nlr = nall // nhf
        nal = nlr * nhf

        # print('time:',time1[0],'##',S_time[0:20]-time1[0])
        # print('time:',time1[nlr-1],'##',S_time[(nlr-1)*nhf:nlr*nhf]-time1[nlr-1])
        S_time = np.reshape(S_time[0:nal], (nlr, nhf))  # warning this reshaping throws away the last few 20 Hz sample
        # print('time shape:',n1,S_time.shape)

        S_swh = np.ma.getdata(S['data_20']['ku'].variables['swh_ocean'][:])  # this is MLE4
        swh1 = np.reshape(S_swh[0:nal], (nlr, nhf))

        S_lat = np.ma.getdata(S['data_20'].variables['latitude'][:])
        lat1 = np.reshape(S_lat[0:nal], (nlr, nhf))

        S_lon = np.ma.getdata(S['data_20'].variables['longitude'][:])
        lon1 = np.reshape(S_lon[0:nal], (nlr, nhf))

        S_waveform = np.ma.getdata(S['data_20']['ku'].variables['power_waveform'][:])
        waveforms = np.reshape(S_waveform[0:nal, :], (nlr, nhf, np.shape(S_waveform)[1]))

        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission in ['saral', 'altika']:
        # example file='SRL_GPS_2PfP001_0641_20130405_141055_20130405_150113.CNES.nc'
        swh2 = np.ma.getdata(S.variables['swh_40hz'][:])  # this is MLE4?
        lat2 = np.ma.getdata(S.variables['lat_40hz'][:])
        lon2 = np.ma.getdata(S.variables['lon_40hz'][:])
        time2 = np.ma.getdata(S.variables['time_40hz'][:])
        time1 = np.ma.getdata(S.variables['time'][:])
        off2 = np.ma.getdata(S.variables['off_nadir_angle_pf'][:])
        waveform = np.ma.getdata(S.variables['waveforms_40hz'][:])

        timeref = "2000-01-01 00:00:00.0"  # WARNING: this should be read from the attribute of the time variable ...

    if mission.lower() in ['cfosat']:
        # example file='CFO_OP07_SWI_L2_____F_20241221T043021_20241221T054444.nc'
        S = netCDF4.Dataset(filename, 'r')
        swh1 = np.ma.getdata(S.variables['nadir_swh_1Hz'][:])
        #    swh_sgdr2 = np.ma.getdata(S.variables['nadir_swh_native'][:])
        waveforms = np.ma.getdata(S.variables['waveforms_40hz'][:])
        lat1 = np.ma.getdata(S.variables['lat_nadir_1Hz'][:])
        lon1 = np.ma.getdata(S.variables['lon_nadir_1Hz'][:])
        #        time1 = np.ma.getdata(S.variables['time_nadir_1Hz'][:])
        time1 = S.variables['time_nadir_1Hz'][:]
        flag1 = np.ma.getdata(S.variables['flag_valid_swh_1Hz'][:])
        timeref = "2009-01-01 00:00:00 0:00"
    # reference_time = pd.Timestamp("2014-09-05")
    ds = xr.Dataset(
        {
            "swh2d": (["time", "meas_ind"], swh1),
            "lon2d": (["time", "meas_ind"], lon1),
            "lat2d": (["time", "meas_ind"], lat1),
            "flag2d": (["time", "meas_ind"], flag1),
            "waveforms": (["time", "meas_ind", "wvf_ind"], waveforms),
        },
        coords={
            "time": time1[0:nlr],
            "reference_time": timeref,
        },
    )
    return ds
