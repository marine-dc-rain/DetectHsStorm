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
def  alti_paths(mission)  :

# Default values for the Jason series 
    tau=3.125 #gate spacing in ns
    SigmaP=0.513*tau
    nump=90
    total_gate_number=104
    nominal_tracking_gate=31     

    if mission.lower() in ['cfosat']:
         PATH_ALTI_in = '/home/datawork-cersat-public/provider/cnes/satellite/l2/cfosat/swim/swi_l2____/op??/*/YYYY/*/*F_YYYYMM*'
         TAG_ALTI='GDR_CFO'
    return PATH_ALTI_in, TAG_ALTI

######################  Generic reading of altimeter data: retracked parameters
def  alti_read_l2lr(mission,filename):
    if mission.lower() in ['cfosat']:
    # example file='CFO_OP07_SWI_L2_____F_20241221T043021_20241221T054444.nc'
        S = netCDF4.Dataset(filename, 'r')
        swh_sgdr1 = np.ma.getdata(S.variables['nadir_swh_1Hz'][:])
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
        {   "swh_1hz": (["time"], swh_sgdr1),
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



def get_storm_info_from_savemap(ds):
    '''
    function to be applied by a "MAP" function of xarray
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

##########################################################"
# def get_storm_by_timestep(ds1,levels,Npix_min,amp_thresh, d_thresh_min, d_thresh_max, min_area,  area_forgotten_ratio, plot_output = False, plot_example = False):
def get_storm_by_file(mission,filename,yy,mm,hs_thresh,min_len, count0=0,plot_output = False, plot_example = False):   
    # --- concat [-180: 360] ----

    res=[]
    ds3=[]
    count1=count0
    try:
   #for iloop in [0]:
        ds=alti_read_l2lr(mission,filename)
    except:
        print('Problem in file:',filename)  
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


