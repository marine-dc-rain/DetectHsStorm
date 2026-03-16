#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import sys
PATH_with_functions='/'
sys.path.append(PATH_with_functions)

import os
import pathlib

import numpy as np
import scipy as sp
import glob
import pandas as pd
import datetime as dtm
import xarray as xr
import matplotlib as mpl
import multiprocessing as mp

import cartopy.crs as ccrs
import cartopy.feature as cfeature

disttocoast = xr.open_dataarray('distance2coast180.nc')

PATH = '/home1/datawork/mdecarlo/TBH_Model/'

# ==================================================================================
# === 1. Functions tracking ========================================================
# ==================================================================================

def py_files(root,suffix='.nc'): 
    """Recursively iterate all the .nc files in the root directory and below""" 
    for path, dirs, files in os.walk(root):
        if suffix[0]=='.':
            yield from (os.path.join(path, file) for file in files if pathlib.Path(file).suffix == suffix)
        else:
            yield from (os.path.join(path, file) for file in files if file[-len(suffix):] == suffix)
            
def haversine(lat1, lon1, lat2, lon2):
    ''' This code is contributed
    by ChitraNayal
    from https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
    distance between latitudes
    and longitudes'''
    dLat = (lat2 - lat1) * np.pi / 180.0
    dLon = (lon2 - lon1) * np.pi / 180.0

    # convert to radians
    lat1 = lat1 * np.pi / 180.0
    lat2 = lat2 * np.pi / 180.0

    # apply formulae
    a = (pow(np.sin(dLat / 2), 2) +
         pow(np.sin(dLon / 2), 2) *
             np.cos(lat1) * np.cos(lat2));

    rad = 6371
    c = 2 * np.arcsin(np.sqrt(a))
    return rad * c
           
# -- function CROSSES_LAND ------------------------
def crosses_land(disttocoast,lat1s,lon1s,lat2s,lon2s,Num_steps=250,is2D=1):
    ''' Function to determine if there is a continent between 2 points.
    with disttocoast obtained from \n 
    disttocoast = xr.open_dataarray('distance2coast.nc')
    '''
    import cartopy.geodesic as cgeo
    if (len(lat1s)>1)&(len(lat2s)>1):
        if len(lat1s) == len(lat2s):
            print('Considering pairs of points')
        else:
            print('Error : lat1s and lat2s should be same size or one should be scalar')
            return
    elif (len(lat1s)>1)&(len(lat2s)==1):
        lat2s = np.full_like(lat1s,lat2s)
        lon2s = np.full_like(lat1s,lon2s)
    elif (len(lat2s)>1)&(len(lat1s)==1):
        lat1s = np.full_like(lat2s,lat1s)
        lon1s = np.full_like(lat2s,lon1s)
    else:
        lat1s = np.array(lat1s).reshape(1)
        lat2s = np.array(lat2s).reshape(1)
        lon1s = np.array(lon1s).reshape(1)
        lon2s = np.array(lon2s).reshape(1)
        
    if is2D:
        lons2_v1 = np.where((lon1s-lon2s)>180,lon2s+360,lon2s)
        lons1_v1 = np.where((lon1s-lons2_v1)<-180,lon1s+360,lon1s)
        lons = np.linspace(lons1_v1,lons2_v1,Num_steps)
        lats = np.linspace(lat1s,lat2s,Num_steps)
        
        lonxx = xr.DataArray((lons+180)%360-180, dims=("llx","np"), coords={"llx":np.arange(Num_steps)})
        latxx = xr.DataArray(lats, dims=("llx","np"), coords={"llx":np.arange(Num_steps)})
        
        tointer = xr.Dataset({'lon': lonxx,'lat': latxx})
        
        D = disttocoast.interp(lon=tointer.lon,lat=tointer.lat,kwargs={'fill_value':np.nan}).data
        isCrossLand = np.any(np.isnan(D),axis=0)
    else:
        isCrossLand = np.zeros(np.shape(lat1s),dtype=np.bool)
        for ia, lat1 in enumerate(lat1s):
            lon1 = lon1s[ia]
            lon2 = lon2s[ia]
            lat2 = lat2s[ia]
            if (lon1-lon2)>180:
                lon2 = lon2 + 360
            elif (lon1-lon2)<-180:
                lon1 = lon1 + 360
                
            lons = np.linspace(lon1,lon2,Num_steps)
            lats = np.linspace(lat1,lat2,Num_steps)

            lonxx = xr.DataArray((lons+180)%360-180, dims=("llx"), coords={"llx":np.arange(Num_steps)})
            latxx = xr.DataArray(lats, dims=("llx"), coords={"llx":np.arange(Num_steps)})

            tointer = xr.Dataset({'lon': lonxx,'lat': latxx})

            D = disttocoast.interp(lon=tointer.lon,lat=tointer.lat,kwargs={'fill_value':np.nan}).data
            isCrossLand[ia] = np.any(np.isnan(D))
    return isCrossLand


def function_1_file(file_ww3,file_era5,disttocoast,threshold_dist=400):
    dsW = xr.open_dataset(file_ww3)
    dsE = xr.open_dataset(file_era5)

    NSW = dsW.numStorm.compute().values
    NSE = dsE.numStorm.compute().values

    dist_T = dsW.time.compute().values.reshape(-1,1) - dsE.time.compute().values.reshape(1,-1)
    dist_TBN = dist_T==np.timedelta64(0,'ns')

    ind0,ind1 = np.where(dist_TBN)
    distF = np.full(dist_TBN.size, np.inf)

    loW = dsW.lon_max.compute().values
    laW = dsW.lat_max.compute().values
    loE = dsE.lon_max.compute().values
    laE = dsE.lat_max.compute().values

    croLand = crosses_land(disttocoast,laW[ind0],loW[ind0],laE[ind1],loE[ind1],Num_steps=50)

    indcross = np.where(croLand==False)[0]
    indnew = np.ravel_multi_index((ind0[indcross],ind1[indcross]),np.shape(dist_TBN))

    dist0 = haversine(laW[ind0[indcross]],loW[ind0[indcross]],laE[ind1[indcross]],loE[ind1[indcross]])
    ind_distBN = np.where(dist0<threshold_dist)[0]
    distF[indnew[ind_distBN]] = dist0[ind_distBN]

    dist = distF.reshape(dist_TBN.shape)
    
    ind_storm_equiv_ERA5 = np.zeros(dsW.sizes['x'])-1
    ind_storm_equiv_WW3 = np.zeros(dsE.sizes['x'])-1
    
    X_1 = np.arange(np.size(dist_TBN,0))
    X_2 = np.arange(np.size(dist_TBN,1))
    # ------ For WW3 : GET ind equivalent ERA5 --------------------
    # -- Take min along WW3 axis (i.e. which WW3 storm is closest to each ERA5 storm)
    i1 = np.argmin(dist,axis=0)
    dist2F = np.full(dist_TBN.size, np.inf)
    # -- Only keep the distance corresponding to smaller distance over WW3 axis
    ii1 = np.ravel_multi_index((i1,X_2),np.shape(dist_TBN))
    dist2F[ii1] = distF[ii1]
    dist2 = dist2F.reshape(dist_TBN.shape)
    # -- Take min along ERA5 axis -----
    i12 = np.argmin(dist2,axis=1)
    indmin12 = np.where(np.isfinite(np.min(dist2,axis=1)))[0]

    ind_storm_equiv_ERA5[indmin12] = NSE[i12[indmin12]]

    # ------ For ERA5 : GET ind equivalent WW3 --------------------
    # -- Take min along ERA5 axis (i.e. which ERA5 storm is closest to each WW3 storm)
    i2 = np.argmin(dist,axis=1)
    dist2F = np.full(dist_TBN.size, np.inf)
    # -- Only keep the distance corresponding to smaller distance over ERA5 axis
    ii2 = np.ravel_multi_index((X_1,i2),np.shape(dist_TBN))
    dist2F[ii2] = distF[ii2]
    dist2 = dist2F.reshape(dist_TBN.shape)
    # -- Take min along WW3 axis -----
    i22 = np.argmin(dist2,axis=0)
    indmin22 = np.where(np.isfinite(np.min(dist2,axis=0)))[0]

    ind_storm_equiv_WW3[indmin22] = NSW[i22[indmin22]]

    dsE1 = dsE.assign(ind_storm_WW3=('x',ind_storm_equiv_WW3))
    dsW1 = dsW.assign(ind_storm_ERA5=('x',ind_storm_equiv_ERA5))

    dsW1.to_netcdf(file_ww3.replace('.nc','_v2.nc'))
    dsE1.to_netcdf(file_era5.replace('.nc','_v2.nc'))
    
    return True
    
# ==================================================================================
# === 2. Main ======================================================================
# ==================================================================================
list_ww3 = list(py_files(PATH+'WW3_LOPS/tracking/transi/',suffix='.nc'))

pool = mp.Pool(7)

results = pool.starmap_async(function_1_file,[(f,f.replace('WW3_LOPS','ERA5'),disttocoast) for f in list_ww3]).get()
print(results)
pool.close()






