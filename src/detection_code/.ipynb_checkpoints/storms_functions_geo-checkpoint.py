#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import sys
import os
import pathlib

import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
'''
Functions :
- lat_lon_cell_area(lat_lon_grid_cell) => area_km2 (with lat_lon_grid_cell = west, south, east, north)
- haversine(lat1, lon1, lat2, lon2) => distance between 2 points on sphere (km)
- spatial_filter(field, res, cut_lon, cut_lat)
- distance_matrix(lons,lats) => distance (km) in a matrix (same as haversine in matrix)
- crosses_land(disttocoast,lat1,lon1,lat2,lon2) => bool : is there land between (lat1,lon1) and (lat2,lon2) ?
'''


def lat_lon_cell_area(lat_lon_grid_cell):
    
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    AVG_EARTH_RADIUS_METERS = 6371008.8
    
    west, south, east, north = lat_lon_grid_cell
    
    west = (west)*np.pi/180.
    east = (east)*np.pi/180.
    south = (south)*np.pi/180.
    north = (north)*np.pi/180.
    
    return (east - west) * (np.sin(north) - np.sin(south)) * (AVG_EARTH_RADIUS_METERS**2)
    
def haversine(lat1, lon1, lat2, lon2):
    # This code is contributed
    # by ChitraNayal
    # from https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
    # distance between latitudes
    # and longitudes
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
    
    
def spatial_filter(field, res, cut_lon, cut_lat):
    '''
    Performs a spatial filter, removing all features with
    wavelenth scales larger than cut_lon in longitude and
    cut_lat in latitude from field (defined in grid given
    by lon and lat).  Field has spatial resolution of res
    and land identified by np.nan's
    '''
    shp_field = field.shape
    field_filt = np.zeros(shp_field)

    # see Chelton et al, Prog. Ocean., 2011 for explanation of factor of 1/5
    sig_lon = (cut_lon/5.) / res
    sig_lat = (cut_lat/5.) / res

    land = np.isnan(field.flatten())
    field = field.flatten()
    field[land] = np.nanmean(field)
    
    field = np.reshape(field,shp_field)

    # field_filt = field - ndimage.gaussian_filter(field, [sig_lat, sig_lon])
    field_filt = ndimage.gaussian_filter(field, [sig_lat, sig_lon]).flatten()
    field_filt[land] = np.nan

    return np.reshape(field_filt,shp_field)


def distance_matrix(lons,lats):
    '''Calculates the distances (in km) between any two cities based on the formulas
    c = sin(lati1)*sin(lati2)+cos(longi1-longi2)*cos(lati1)*cos(lati2)
    d = EARTH_RADIUS*Arccos(c)
    where EARTH_RADIUS is in km and the angles are in radians.
    Source: http://mathforum.org/library/drmath/view/54680.html
    This function returns the matrix.'''

    EARTH_RADIUS = 6378.1
    X = len(lons)
    Y = len(lats)
    assert X == Y, 'lons and lats must have same number of elements'

    d = np.zeros((X,X))

    #Populate the matrix.
    for i2 in range(len(lons)):
        lati2 = lats[i2]
        loni2 = lons[i2]
        c = np.sin(np.radians(lats)) * np.sin(np.radians(lati2)) + \
            np.cos(np.radians(lons-loni2)) * \
            np.cos(np.radians(lats)) * np.cos(np.radians(lati2))
        d[abs(c)<1,i2] = EARTH_RADIUS * np.arccos(c[abs(c)<1])

    return d
    
def crosses_land(disttocoast,lat1,lon1,lat2,lon2):
    Num_steps = 250
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

    return np.any(np.isnan(D))
