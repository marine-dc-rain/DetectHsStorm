#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==================================================================================
# === 0. Import Packages ===========================================================
# ==================================================================================
import sys
import os
import pathlib
import logbook
import numpy as np
import pandas as pd
import xarray as xr


from detection_code.storms_functions_geo import haversine, crosses_land

'''
Functions:
- one_storm_vs_old_storms(ds1st,dsold,threshold_dist,disttocoast) => isLinked_to_previous (bool), X_previous (int, name of previous), dist_to_prev (int, km) (internal)
- track_for_1_file(filename,disttocoast,PATHsave,threshold_dist = 400)
- track_for_1_transition(PATH,f01,f02,disttocoast,threshold_dist = 400)
'''
# ==================================================================================
# === 1. Functions tracking ========================================================
# ==================================================================================
def one_storm_vs_old_storms(ds1st,dsold,threshold_dist,disttocoast):
	dist = haversine(ds1st['lat_max'],ds1st['lon_max'],dsold['lat_max'],dsold['lon_max']).compute().data
	if len(dist) > 0:
	# get new storm closest to old one
		imin = np.argmin(dist)
	# get all new storms close enough to old one 
		ind_potential_points = np.where(dist < threshold_dist)[0]
	# Flag if the closest storm is separated from old by land ...
		croLand = crosses_land(disttocoast,ds1st['lat_max'].data,ds1st['lon_max'].data, dsold.isel(x=imin)['lat_max'].data,dsold.isel(x=imin)['lon_max'].data)
	# if closest storm cannot be (land in between..) go to other storms close enough
		while (len(ind_potential_points)>0)&(croLand):
			dist[imin] = np.inf
			imin = np.argmin(dist)
			ind_potential_points = np.where(dist < threshold_dist)[0]
			croLand = crosses_land(disttocoast,ds1st['lat_max'].data,ds1st['lon_max'].data,dsold.isel(x=imin)['lat_max'].data,dsold.isel(x=imin)['lon_max'].data)

		if len(ind_potential_points)>0 : # get the closest point 
			potential_point = dsold.isel(x=imin)
			isLinked_to_previous = True
			X_previous = potential_point['x'].values
			dist_to_prev = dist[imin]
		else:
			isLinked_to_previous = False
			X_previous = -1
			dist_to_prev = np.inf
	else:
		isLinked_to_previous = False
		X_previous = -1
		dist_to_prev = np.inf
		
	return isLinked_to_previous, X_previous, dist_to_prev
		   
# ---- Fonction to track (= link storms for time T to time T+1) ---------------------
def track_for_1_file(fin,fout,PATH_in, PATH_out, disttocoast, log,threshold_dist = 400):#(filename,disttocoast,PATHsave,threshold_dist = 400):
	#print(filename)
	all_storms = xr.open_dataset(os.path.join(PATH_in,fin))
	all_storms = all_storms.assign_coords(x = (['x'],np.arange(all_storms.sizes['x'])))

	Tsteps = np.unique(all_storms.time)
	Tstepsnum = len(Tsteps)
	counts = all_storms.sizes['x']
	countST = 0
	numStorm = np.full((counts),fill_value=-1)

	for k,tk in enumerate(Tsteps):# ----- LOOP over time steps -------------
		# ---- get storms that were detect during time step -----------
		if k%20 == 0:
			log.info('Processing time step '+str(k)+' over '+str(Tstepsnum))
		ind = np.where(all_storms['time'] == tk)[0]
		if k == 0:
		# ---- if first time step : all storms are new (to be modified after during 'transitions')
			for kst in range(len(ind)):
				numStorm[kst] = kst
				countST = countST+1
		else:
		# ---- if not first time step : save previous storms as dsold ---
			dsold = all_storms.isel(x=np.where(all_storms['time']==Tsteps[k-1])[0])
			dssel = all_storms.isel(x=ind) # --- dssel  is the dataset associated with the storms in the current  time step
			isLinked_to_previous = np.zeros((len(ind)),dtype=bool)
			X_previous = np.zeros((len(ind)),dtype=int)
			dist = np.zeros((len(ind)))
			X_nows = np.zeros((len(ind)),dtype=int)
			
			for kst in range(len(ind)): # --- Loop over the storms 
				ds1st = dssel.isel(x=kst)
				X_nows[kst] = ds1st['x']
				isLinked_to_previous[kst],X_previous[kst],dist[kst] = one_storm_vs_old_storms(ds1st,dsold,threshold_dist,disttocoast)
				
			# deal with case when 2 new storms refer to same old
			indlinked = np.where(isLinked_to_previous)[0]
			Xu, countsu = np.unique(X_previous[indlinked],return_counts=True)
			if np.any(countsu>1):
			# if len different: some X_previous are double
				for Dble_st in Xu[countsu>1]:
					ind_dbl = np.where(X_previous == Dble_st)[0]
					indimin = np.argmin(dist[ind_dbl])
					indnomin = np.setdiff1d(np.arange(0,len(ind_dbl)),indimin)

					isLinked_to_previous[ind_dbl[indnomin]] = False
					X_previous[ind_dbl[indnomin]] = -1
					dist[ind_dbl[indnomin]] = np.inf
			
			for kst in range(len(ind)): # --- Loop over the storms   
				
				if isLinked_to_previous[kst]: # --- the closest point exists
					numStorm[X_nows[kst]] = numStorm[X_previous[kst]]
				else: # --- if no close point : This is a new storm !
					numStorm[X_nows[kst]] = countST
					countST = countST+1

	all_storms = all_storms.assign_coords({"numStorm":(("x"),numStorm)})
	all_storms.to_netcdf(os.path.join(PATH_out,'__'+fout))

# ---- Fonction to track for transitions (= link storms for time Tend of file F to time T1 of file F+1) ----------
def track_for_1_transition(PATH,f01,f02,disttocoast,threshold_dist = 400):
	f1 = os.path.join(PATH,f01)
	f2 = os.path.join(PATH,'__'+f02)
	all_storms_m1 = xr.open_dataset(f1)
	all_storms_m2 = xr.open_dataset(f2)

	# -- first time step of file2 -------------------
	T01 = pd.to_datetime(all_storms_m2.isel(x=0).time.item())
	# -- last time step of file1 -------------------
	T0 = T01- pd.Timedelta('3h')

	# --- select storms associated to those timesteps only -----
	dsold = all_storms_m1.where(all_storms_m1.time==T0,drop=True)
	dssel = all_storms_m2.where(all_storms_m2.time==T01,drop=True)

	# --- get maximum name of storm for file1 (will become new zero of file 2) ----
	Nstorms_old = all_storms_m1.numStorm.max().item()
	# -- name of storms for file2 ---------------------		
	numStorm2 = all_storms_m2.numStorm.data
	
	isLinked_to_previous = np.zeros((dssel.sizes['x']),dtype=bool)
	X_previous = np.zeros((dssel.sizes['x']),dtype=int)
	dist = np.zeros((dssel.sizes['x']))
	X_nows = np.zeros((dssel.sizes['x']),dtype=int)
				
	for kst in range(dssel.sizes['x']):
		ds1st = dssel.isel(x=kst)
		X_nows[kst] = ds1st['numStorm'].item()
		isLinked_to_previous[kst], X_previous[kst], dist[kst] = one_storm_vs_old_storms(ds1st,dsold,threshold_dist,disttocoast)
	
	# deal with case when 2 new storms refer to same old
	indlinked = np.where(isLinked_to_previous)[0]
	Xu, countsu = np.unique(X_previous[indlinked],return_counts=True)
	if np.any(countsu>1):
	# if len different: some X_previous are 2ble
		for Dble_st in Xu[countsu>1]:
			ind_dbl = np.where(X_previous == Dble_st)[0]
			indimin = np.argmin(dist[ind_dbl])
			indnomin = np.setdiff1d(np.arange(0,len(ind_dbl)),indimin)

			isLinked_to_previous[ind_dbl[indnomin]] = False
			X_previous[ind_dbl[indnomin]] = -1
			dist[ind_dbl[indnomin]] = np.inf
	
	for kst in range(dssel.sizes['x']): # --- Loop over the storms   
		if isLinked_to_previous[kst]:
			old_name_of_storm = X_nows[kst]
			numStorm2[numStorm2==old_name_of_storm] = all_storms_m1.numStorm.sel(x=X_previous[kst]) - (Nstorms_old+1)
			
	# --- at this point : all storms that already existed in the previous year have a negative numStorm2
	# --- get all 'names' of the storms that only exist on year2 ---
	nsto_u = np.unique(numStorm2[numStorm2>=0])
	# --- rename these storms from 0 to N on a linear way (no jumps) ---
	numStorm3 = np.zeros_like(numStorm2)
	for countST,ist in enumerate(nsto_u): # --- Loop for all storms only in year2
		ind = np.where(numStorm2==ist)[0]
		if len(ind)>0:
			# --- change previous labels to linear increase --- 
			numStorm3[ind] = countST
			countST = countST
	# --- for negative values of numStorm2 (that already existed in previous year), assign the negative value to the new labeling		
	numStorm3[numStorm2<0] = numStorm2[numStorm2<0]
	# --- then, add offset (number of last storm from previous year) to have a continuous labeling ---
	numStorm3 = numStorm3 + (Nstorms_old+1)
			
	all_storms_m2 = all_storms_m2.assign_coords({"numStorm":(("x"),numStorm3)})
	# --- change x value to be able to concatenante afterwards
	x_old = all_storms_m1['x'].max().item()
	all_storms_m2 = all_storms_m2.assign_coords({"x":(("x"),np.arange(all_storms_m2.sizes['x']) + x_old+1)})

	all_storms_m2.to_netcdf(os.path.join(PATH,f02))
