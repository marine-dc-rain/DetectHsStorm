#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import os
import sys
import numpy as np
import scipy.ndimage as ndimage
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from detection_code.storms_functions_geo import distance_matrix, spatial_filter

'''
Functions:
- rename_duplicates(regions,NX) : function (internal) to deal with duplication of regions of interest generated while dealing with the -180/180 issue.
- get_storm_info_from_savemap(ds) : internal function to be applied in get_storm_by_timestep
- get_storm_by_timestep(ds1,levels,amp_thresh, min_area, area_forgotten_ratio, plot_output = False, plot_example = False):   
'''

def rename_duplicates(regions,NX):
    map_m1800 = np.copy(regions[:,:NX//2])
    map_0180 = np.copy(regions[:,NX//2:NX])
    map_180360 = np.copy(regions[:,NX:])
    map_m180180 = np.copy(regions[:,:NX])

    uregions_m1800 = np.unique(map_m1800)
    uregions_0180 = np.unique(map_0180)
    uregions_180360 = np.unique(map_180360)
    
    # --- get labels that intersect between regions ---- 
    intersect_at_0 = [r for r in uregions_m1800 if (r in uregions_0180)&(r>-1)]
    intersect_at_180 = [r for r in uregions_180360 if (r in uregions_0180)&(r>-1)]

    # create a mask to protect the area of intersection around 0
    BN_inters_0 = np.zeros_like(map_m180180,dtype=bool)
    _BN_inters_180 = np.zeros_like(map_180360,dtype=bool)
    BN_inters_180 = np.zeros_like(map_m180180,dtype=bool)
    BN_inters_tot = np.zeros_like(regions,dtype=bool)
    for r in intersect_at_0:
        BN_inters_0[map_m180180 == r] = True
        BN_inters_tot[regions == r] = True
        
    for r in intersect_at_180:
        _BN_inters_180[map_180360 == r] = True
        BN_inters_tot[regions == r] = True

    BN_inters_180[:,:np.size(map_180360,1)] = _BN_inters_180
    
    regions_orig = np.copy(map_m180180)
    regions_orig2 = np.copy(map_m180180)
    # -- create duplicate map : same size as regions_orig (-180/180) with values from 180/360 (and -100 to complete)
    regions_dupl0 = np.zeros_like(map_m180180)-100.
    regions_dupl0[:,:np.size(map_180360,1)] = np.copy(map_180360)

    regions_dupl2 = np.zeros_like(map_m180180)-100.
    # m0 = np.copy(map_180360)
    # m0[BN_inters_180==False] = -100
    regions_dupl2[:,:np.size(map_180360,1)] = np.copy(map_180360)
    regions_dupl2[BN_inters_180==False] = -100
    
    # if intersectin at 0 = True : change name to original
    
    regions_dupl0[BN_inters_0==True] = map_m180180[BN_inters_0==True]
    # regions_dupl2[BN_inters_0==True] = -100
    
    # -- modify regions origin, to give name of duplicated
    # -- only if duplicate exist and no intersection at 0
    regions_orig[(regions_dupl0 > -1)&(BN_inters_0==False)] = regions_dupl0[(regions_dupl0 > -1)&(BN_inters_0==False)]
    regions_orig2 = np.copy(regions_orig) #[(regions_dupl2 > -1)&(BN_inters_0==False)] = -100
    regions_orig2[BN_inters_180==True] = -100
    # --- get same name to same regions --- 
    regions_newname = np.concatenate([regions_orig,regions_dupl0[:,:np.size(map_180360,1)]],axis=1)
    regions_only_big_area = np.concatenate([regions_orig2,regions_dupl2[:,:np.size(map_180360,1)]],axis=1)
    
    regions_interOnly = np.copy(regions_newname)
    regions_interOnly[BN_inters_tot==False] = -100
        
    return regions_newname, regions_only_big_area ,regions_interOnly
    
def get_storm_info_from_savemap(ds):
    '''
    function to be applied by a "MAP" function of xarray
    '''
    idm = ds.hs.idxmax()
    arsum = ds.areakm2.sum()
    hsmax = ds.hs.max()

    dsn = xr.Dataset({'lon_max' : (idm.longitude+180)%360 -180, 
                      'lat_max' : idm.latitude, 
                      'time' : ds.time, 
                      'hs_max' : hsmax,
                      'areastorm' : arsum, 
                     }
                    )
    return dsn

# def get_storm_by_timestep(ds1,levels,Npix_min,amp_thresh, d_thresh_min, d_thresh_max, min_area,  area_forgotten_ratio, plot_output = False, plot_example = False):
def get_storm_by_timestep(ds1,levels,amp_thresh, min_area, area_forgotten_ratio, plot_output = False, plot_example = False):   
    # --- concat [-180: 360] ----
    ds2 = ds1.copy(deep=True).sel(longitude=slice(None,0))  # OK for WW3 that starts at -180 but ERA5 ??? 
    ds2['longitude'] = ds2['longitude']+360.
    dsTot = xr.concat((ds1,ds2),dim='longitude')
    if plot_example:
        plt.figure(figsize=(12,5))
        im = plt.pcolormesh(dsTot.longitude,dsTot.latitude,dsTot.hs,cmap='jet')
        plt.colorbar(im)
        plt.title('Hs after concatenation (map from -180 to 360)')
    
    swh_filt0 = spatial_filter(dsTot['hs'].data, 0.5, 4., 4.)
    swh_filt = dsTot['hs'].copy(data=swh_filt0)
        
    field20 = swh_filt

    area2 = dsTot['areakm2'].data
    field20 = field20.where(~np.isnan(field20),0)
    field2 = field20.data

    NX = ds1.sizes['longitude']

    # llon2,llat2 = np.meshgrid(dsTot['longitude'],dsTot['latitude'])

    regions_old = np.zeros_like(field2.data,dtype='int')-100
    to_save = np.zeros_like(field2.data,dtype='int')-100
    to_save_intersec = np.zeros_like(field2.data,dtype='int')-100
    
    countst = 0
    if plot_example:
        fig,axs = plt.subplots(3,2,figsize=(16,12))
        level_selec = 21
        

    for ilev, lev in enumerate(levels):
        # --- 1. Find all regions with Hs greater than ilev  ---------------
        regions, nregions = ndimage.label( (field2 > lev).astype(int) )
        regions[regions==0] =-100
        
        regions_new, regions_only_big_area, regions_interOnly = rename_duplicates(regions,NX)
        uregions = np.unique(regions_new[regions_new>-100])
        
        if plot_example:
            if ilev == level_selec:
                ax = axs[0,0]
                im = ax.pcolormesh(dsTot.longitude,dsTot.latitude,regions,cmap='jet',vmin=-1)
                plt.colorbar(im,ax=ax)
                ax.set_title('labelisation step, num lev = '+str(ilev)+', hs lim = '+str(lev))
                ax = axs[0,1]
                im = ax.pcolormesh(dsTot.longitude,dsTot.latitude,regions_new,cmap='jet',vmin=-1,vmax=regions_new.max())
                plt.colorbar(im,ax=ax)
                ax.set_title('renamed regions ')
                ax = axs[1,0]
                im = ax.pcolormesh(dsTot.longitude,dsTot.latitude,regions_interOnly,cmap='jet',vmin=-1,vmax=regions_new.max())
                plt.colorbar(im,ax=ax)
                ax.axvline(0,ls='--',color='m')
                ax.axvline(180,ls='--',color='m')
                ax.set_title('regions at intersection (0° and 180°)')
                ax = axs[1,1]
                im = ax.pcolormesh(dsTot.longitude,dsTot.latitude,regions_only_big_area,
                                   cmap='jet',vmin=-1,vmax=regions_new.max())
                plt.colorbar(im,ax=ax)
                ax.set_title('unique regions with only big area')
                ax = axs[2,0]
                im = ax.pcolormesh(dsTot.longitude,dsTot.latitude,(to_save_intersec>-1),
                                   cmap='grey',vmin=0,vmax=1)
                plt.colorbar(im,ax=ax)
                ax.set_title('already saved at previous step')
                ax = axs[2,1]
                im = ax.pcolormesh(dsTot.longitude,dsTot.latitude,(to_save_intersec<-1)*(regions_new >-1),#+1)*(regions_new >-1),
                                   cmap='grey',vmin=0,vmax=1)
                plt.colorbar(im,ax=ax)
                ax.set_title('regions - (already saved)')
                

        for iir, ir in enumerate(uregions):
            # ---- 
            regionNB = (regions_new == ir)
            regionNB_nodupl = (regions_only_big_area == ir)
            # == regions_new contains duplicates ===
            is_already_saved = np.any((to_save_intersec[regionNB]>-1))
            u_regions_saved_in = np.unique((to_save_intersec[(regionNB) & (to_save_intersec>-1)]))
            u_old_regions_in0 = np.unique((regions_old[(to_save_intersec<-2) & (regionNB) & (regions_old>-1)])) # regions that are not saved
            u_old_regions_out0 = np.unique(regions_old[(regionNB) & (to_save_intersec>-1)])
            u_old_regions_in = np.setdiff1d(u_old_regions_in0, u_old_regions_out0)

            if is_already_saved: # ---- inside the region, there is already a save storm:
                # --- case : there was also another "storm" detected at previous level that did not match all required flags for saving
                # -- get area of saved storms ---------
                area_old_max = 0
                for u_sav_in in u_regions_saved_in:
                    area_u = np.sum(area2[to_save == u_sav_in])
                    area_old_max = np.max((area_old_max,area_u))
                # --- for the forgotten storm, if area big enough : save -----     
                if len(u_old_regions_in) > 0:
                    for u_old in u_old_regions_in: # --- loop over "forgotten storm" 
                        region_old_u_old = regionNB_nodupl & (regions_old == u_old)
                        interior = ndimage.binary_erosion(region_old_u_old)
                        exterior = np.logical_xor(region_old_u_old, interior)
                        if interior.sum() == 0:
                            continue
                        area_u_old = np.sum(area2[region_old_u_old])
                        # --- save if not too small ------------------------------
                        if area_u_old >= area_old_max*area_forgotten_ratio:
                            to_save[region_old_u_old] = countst
                            to_save_intersec[region_old_u_old] = countst
                            if np.any(regions_interOnly == ir):
                                to_save_intersec[(regions_old == u_old)] = countst
                            countst = countst + 1
                    # --- end of loop over "forgotten storm" ---------------------

            else: # --- not already saved
                # --- (OLD) Criterion on number of pixel -------------------
                # regionNb_Npix = regionNB_nodupl.astype(int).sum()        
                # eddy_area_within_limits = (regionNb_Npix > Npix_min)
                # --- Criterion on area -------------------
                area_reg = np.sum(area2[regionNB_nodupl])
                eddy_area_within_limits = (area_reg > min_area)
                interior = ndimage.binary_erosion(regionNB_nodupl)
                exterior = np.logical_xor(regionNB_nodupl, interior)

                if interior.sum() == 0:
                    continue

                has_internal_max = np.max(field2[interior]) > field2[exterior].max()
                if np.logical_not(has_internal_max):
                    continue

                amp = (field2[interior].max() - field2[exterior].mean()) / field2[interior].max()
                is_tall_storm = (amp >= amp_thresh)
                if np.logical_not(is_tall_storm):
                    continue

                # Condition of 'shape' (if ellipse :compute both semi-axes)
                # lon_ext = llon2[exterior]
                # lat_ext = llat2[exterior]
                # d = distance_matrix(lon_ext, lat_ext)

                # is_large_enough = np.logical_and((d.max() > d_thresh_min),(d.max() < d_thresh_max))

                if eddy_area_within_limits * has_internal_max * is_tall_storm:# * is_large_enough:
                    # ---- if conditions are respected : the storm is stored as a storm.
                    to_save[regionNB_nodupl] = countst
                    to_save_intersec[regionNB] = countst # -- saved with duplicates
                    countst = countst + 1

        regions_old = regions_new.copy() 
    
    ds3=dsTot.assign({'regions' :(('latitude','longitude' ),to_save)})
    g=ds3.where(ds3.regions>-1.,np.nan).groupby('regions')
    res = g.map(get_storm_info_from_savemap).swap_dims({'regions':'x'})
    res = res.assign_coords(storms_by_t = xr.full_like(res.regions,fill_value=len(g),dtype=int))

    if plot_example:
        plt.figure(fig)
        plt.tight_layout()  
        fig_end = plt.figure(figsize=(12,5))
        im = plt.pcolormesh(ds3.longitude,ds3.latitude,to_save_intersec,cmap='jet',vmin=-1)
        plt.colorbar(im)
        plt.plot(res.lon_max,res.lat_max,'ow',markeredgecolor='k')
        plt.title('detected regions')
    if plot_output:
        fig_end2 = plt.figure(figsize=(12,5))
        im = plt.pcolormesh(ds3.longitude,ds3.latitude,dsTot.hs,cmap='jet',vmin=-1)

        plt.contour(ds3.longitude,ds3.latitude,(to_save_intersec>-1)*1,levels=[0.5],colors='m',linewidths=2)
        plt.contour(ds3.longitude,ds3.latitude,(to_save_intersec>-1)*1,levels=[0.5],colors='w',linewidths=1)
        plt.colorbar(im)
        plt.plot(res.lon_max,res.lat_max,'ow',markeredgecolor='k')
        #plt.plot(res.lon_max,res.lat_max,'+m',lw=4)
        plt.title('Hs with detected regions')
        return res, fig_end2
    else:
        return res

