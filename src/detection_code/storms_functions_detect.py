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
- get_storm_info_from_savemap(ds) : internal function to be applied in get_storm_from_model_by_timestep
- get_storm_from_model_by_timestep(ds1,levels,amp_thresh, min_area, area_forgotten_ratio, plot_output = False, plot_example = False):   
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

# def get_storm_from_model_by_timestep(ds1,levels,Npix_min,amp_thresh, d_thresh_min, d_thresh_max, min_area,  area_forgotten_ratio, plot_output = False, plot_example = False):
def get_storm_from_model_by_timestep(ds1,levels,amp_thresh, min_area, area_forgotten_ratio, plot_output = False, plot_example = False):   
    # --- concat [-180: 360] ----
    ds2 = ds1.copy(deep=True).sel(longitude=slice(None,0))
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

def get_storm_from_sat_by_file(filename, args_findpeaks, output = 'track', name_select_stats = None, Hs_bins = None):
    '''
    Function get_storm_from_sat_by_file
    INPUTS : - filename : (whole path) str
             - args_findpeaks : None or dict containing keywords 
                                    'height':      limits of the height of the signal, associated with min and max as tuple
                                    'prominence':  limits of the prominence of the peak, associated with min and max as tuple)
                                    'width':       limits to the width of the peak, MANDATORY to retrieve info on the width of the peak 
                                                (set to (None,None) if no criterion on width to be given)
                                    'rel_height':  relative height at which to measure the width's peak.
                                By default (args_findpeaks set to None), this dictionnary is set to:
                                args_findpeaks = dict(height=(height_thresh,None),
                                                      prominence=(0.6*height_thresh,None),
                                                      width=(5,None),
                                                      rel_height=relative_height)
                                with height_thresh = 10 and relative_height = 0.9
             - output : str amongst ['track','stats','stats_histo'] to define the type of output wanted
                        - track :       return the dataset with only the peaks selected in it
                        - stats :       compute the infos at peak such as Hs_max or the width. 
                                        time, location and others variables from L2 file have to be listed in name_select_stats in order to be selected.
                        - stats_histo : same as stats + histogram of wave height found in the find
             - name_select_stats :  None or dict containing the variables to select at the peak position and their name in the output file.
                                    The name of the variable in the output file is given as the KEY,
                                    while the name of the variable in the input file is given as the associated VALUE
                                    By default, 
                                    name_select_stats = {'lon_max':'lon',
                                                         'lat_max':'lat',
                                                         'time_max':'time',
                                                         'std_Hs_max':'swh_rms',
                                                         'd2c_max':'distance_to_coast',
                                                         'swh_noise':'swh_noise',
                                                         'std_Hs_num_valid':'swh_num_valid'}
            
             - Hs_bins : (optionnal) used only if output is 'stats_histos', these are the bins to compute the wave height histograms
                          By default, Hs_bins = np.arange(0,28,0.2)


    '''
    outputs_valid = ['track','stats','stats_histo']
    if args_findpeaks is None:
        height_thresh = 10
        relative_height = 0.9
        args_findpeaks = dict(height=(height_thresh,None),prominence=(0.6*height_thresh,None),width=(5,None),rel_height=relative_height)
    
    if (name_select_stats is None) & ('stats' in output):
        name_select_stats = {'lon_max':'lon',
             'lat_max':'lat',
             'time_max':'time',
             'std_Hs_max':'swh_rms',
             'd2c_max':'distance_to_coast',
             'swh_noise':'swh_noise',
             'std_Hs_num_valid':'swh_num_valid'}
    if output not in outputs_valid:
        raise ValueError("get_storm_from_sat_by_file: output must be one of %r." % outputs_valid)
    try:
        if (Hs_bins is None) & (output == 'stats_histo'):
            Hs_bins = np.arange(0,28,0.2)

        ds0 = xr.open_dataset(filename).sortby('time')
        if 'distance_to_coast' in ds0.variables:
            ds = ds0.where(np.isfinite(ds0.distance_to_coast),drop=True)
        else:
            ds = ds0
        swh_denoised = ds['swh_denoised'].where(ds.swh_quality==3)
        # swh_denoised = swh_denoised.assign_coords(time=ds.time).interpolate_na(dim='time')
        peakind,props = sps.find_peaks(swh_denoised.data,**args_findpeaks)
        peaksel = []
        if len(peakind)>0:
            if output == 'stats_histo':
                H_ips = []
                H_bases = []
            if 'stats' in output:
                props2 = {}
            if len(peakind)>1:
                inds_save = []
                if 'stats' in output:
                    Hs_max0 = []
                    ind_left = []
                    ind_right = []
                else:
                    inds_save = []
                    counts = []

                for ipic, pic in enumerate(peakind):
                    inddif = np.setdiff1d(np.arange(len(peakind)),ipic)

                    not_inside_bases = (props['left_ips'][ipic] <  props['left_ips'][inddif]) | (props['right_ips'][ipic] >  props['right_ips'][inddif])
                    not_inside_ips = (props['left_bases'][ipic] <  props['left_bases'][inddif]) | (props['right_bases'][ipic] >  props['right_bases'][inddif])
                    
                    if np.all( not_inside_bases &  not_inside_ips):  # not included in any of others => save     
                        peaksel.append(pic)
                        i1 = np.floor(props['left_ips'][ipic]).astype(int)
                        i2 = np.ceil(props['right_ips'][ipic]).astype(int)
                        if 'stats' in output:
                            Hs_max0.append(props['peak_heights'][ipic])
                            ind_left.append(i1)
                            ind_right.append(i2)
                        else:
                            inds_save.append(np.arange(i1,i2))
                            counts.append(np.ones(i2-i1)*(len(peaksel)-1))
                        if output == 'stats_histo':
                            H_ips0,_ = np.histogram(swh_denoised.isel(time=slice(i1, i2)),Hs_bins);
                            H_bases0,_ = np.histogram(swh_denoised.isel(time=slice(props['left_bases'][ipic], props['right_bases'][ipic])),Hs_bins);
                            H_ips.append(H_ips0)
                            H_bases.append(H_bases0)
    
                if output == 'track':
                    inds_save = np.concatenate(inds_save)
                    counts = np.concatenate(counts) 
                
            else: # len(peakind) == 1
                peaksel = peakind 
                i1 = np.floor(props['left_ips'][0]).astype(int)
                i2 = np.ceil(props['right_ips'][0]).astype(int)
                if 'stats' in output:
                    Hs_max0 = props['peak_heights']
                    ind_left = i1
                    ind_right = i2
                else:
                    inds_save = np.arange(i1,i2)
                    counts = np.zeros(i2-i1)
                if output == 'stats_histo':
                    H_ips0,_ = np.histogram(swh_denoised.isel(time=slice(i1, i2)),Hs_bins);
                    H_bases0,_ = np.histogram(swh_denoised.isel(time=slice(props['left_bases'][0], props['right_bases'][0])),Hs_bins);
                    H_ips.append(H_ips0)
                    H_bases.append(H_bases0)
            
            if 'stats' in output:
                Hs_max = xr.DataArray(Hs_max0,dims=['time'])
                ips_xr = xr.DataArray(np.vstack((ind_left,ind_right)),dims=['nbounds','time'])
                above_thresh = xr.Dataset(dict(Hs_max = Hs_max,
                                            ips_xr = ips_xr))
                for var in name_select_stats:
                    above_thresh = above_thresh.assign({var : ds[name_select_stats[var]].isel(time=peaksel)})
                
                if output == 'stats_histo':
                    Hs_bases_xr = xr.DataArray(np.vstack(H_bases),dims=['time','hs_bins'])
                    Hs_ips_xr = xr.DataArray(np.vstack(H_ips),dims=['time','hs_bins'])
                    above_thresh = above_thresh.assign(Histo_Hs_bases=Hs_bases_xr)
                    above_thresh = above_thresh.assign(Histo_Hs_ips=Hs_ips_xr)

                above_thresh = above_thresh.swap_dims({'time':'x'})                           
                above_thresh = above_thresh.assign(filename=("x", np.tile(filename,len(peaksel))))
            else:
                above_thresh = ds.isel(time=inds_save).swap_dims({'time':'x'})
                above_thresh = above_thresh.assign(segment_in_file=("x", counts))
                above_thresh = above_thresh.assign(filename=("x", np.tile(filename,len(inds_save))))

            return above_thresh, filename
    except Exception as inst:
        print('filename :',filename,',', inst,', line number : ',sys.exc_info()[2].tb_lineno)