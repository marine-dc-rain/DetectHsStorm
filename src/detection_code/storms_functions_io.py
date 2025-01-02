import os
import pathlib
import sys

import numpy as np
import xarray as xr

from detection_code.storms_functions_geo import lat_lon_cell_area


'''
Functions :
- py_files(root,suffix='.nc')
- preprocessing_ERA5(ds0)
- read_ERA5_HS_file(Files) : opens multiple ERA5 daily files (preprocessing_ERA5 gives information on their concatenation)
- read_WW3_HS_file(PATH,filename) : open 1 monthly WW3 file + rename variable as needed
'''

def py_files(root,suffix='.nc'): 
	"""Recursively iterate all the .nc files in the root directory and below""" 
	for path, dirs, files in os.walk(root):
		if suffix[0]=='.':
			yield from (os.path.join(path, file) for file in files if pathlib.Path(file).suffix == suffix)
		else:
			yield from (os.path.join(path, file) for file in files if file[-len(suffix):] == suffix)

def preprocessing_ERA5(ds0):
    ds = ds0[['longitude050','latitude050','time','swh']]
    all_lats, all_lons = np.meshgrid(ds0['latitude050'].data, ds0['longitude050'].data, indexing='ij')
    side_length=0.5
    
    lat_lon_grid_cell = np.array([all_lons , all_lats - side_length/2 , all_lons + side_length , all_lats + side_length/2])
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6
    
    ds = ds.rename_dims({'longitude050':'longitude'}).rename_vars({'longitude050':'longitude'})
    ds = ds.rename_dims({'latitude050':'latitude'}).rename_vars({'latitude050':'latitude'})
    ds = ds.rename_vars({'swh':'hs'})
    
    return ds.assign({'areakm2' : (('latitude','longitude'),areakm2)})

def read_ERA5_HS_file(Files):
    DS = xr.open_mfdataset(Files,preprocess=preprocessing_ERA5,concat_dim='time',combine='nested') #,dask='forbidden') #chunks={'time': 10})
    return DS.compute()

def read_WW3_HS_file(PATH,filename):
    print('reading file:',os.path.join(PATH,filename)) 
    ds0 = xr.open_dataset(os.path.join(PATH,filename))
    ds = ds0[['longitude','latitude','time','hs']]
    all_lats, all_lons = np.meshgrid(ds0['latitude'].data, ds0['longitude'].data, indexing='ij')
    side_length=0.5
    
    lat_lon_grid_cell = np.array([all_lons , all_lats - side_length/2 , all_lons + side_length , all_lats + side_length/2])
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6
    
    return ds.assign({'areakm2' : (('latitude','longitude'),areakm2)})

def read_ERA5c_HS_file(PATH,filename):
    ds0 = xr.open_dataset(os.path.join(PATH,filename))
    ds = ds0[['longitude','latitude','time','hs']]
    all_lats, all_lons = np.meshgrid(ds0['latitude'].data, ds0['longitude'].data, indexing='ij')
    side_length=0.5
    
    lat_lon_grid_cell = np.array([all_lons , all_lats - side_length/2 , all_lons + side_length , all_lats + side_length/2])
    areakm2 = lat_lon_cell_area(lat_lon_grid_cell) / 1e6
    
    return ds.assign({'areakm2' : (('latitude','longitude'),areakm2)})

