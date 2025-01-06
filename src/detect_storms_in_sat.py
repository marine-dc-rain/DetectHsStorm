import sys
import glob as glob
import os
import logbook
import shutil

from argparse import ArgumentParser
from enum import IntEnum
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp

from detection_code.storms_functions_io import py_files, read_WW3_HS_file, read_ERA5_HS_file

import detection_code.params_detect as cte
import detection_code.params_altimeters as alti
from detection_code.params_altimeters import get_storm_by_file


logbook.StreamHandler(sys.stdout).push_application()
log = logbook.Logger('StormAnalysis ')

class StepChoice(IntEnum):
	detect_only = 0
	tracking_only = 1
	track_transition_only = 2
	all_from_tracking = 11 # steps  1 + 2
	all_from_detect = 111 # steps 0 + 1 + 2
	all_upto_transition = 110

class StormParser(ArgumentParser):
	def error(self, message):
		log.error(message)
		self.print_help()
		exit(2)
				
class StormDetectionTracking:
	def check_path(self,dir_path):
		if not os.path.isdir(dir_path):
			if os.path.exists(dir_path):
			# exists but is not a directory
				log.error('%s already exists but is not a directory !'% dir_path)
			else:
				os.makedirs(dir_path)
				log.info('%s : Directory created successfully'% dir_path)
		else:
			log.info("%s : Directory already exists!"% dir_path)
			
	def get_all_months(self,year, final_year, month, final_month):
		day_start = pd.Timestamp(year,month,1)
		day_end = pd.Timestamp(final_year,(final_month),1)
		A = pd.date_range(day_start,day_end,freq='MS')
		return np.vstack((A.month,A.year)).T
		
	def run_detection(self,months_years,args):
		ismp = args.is_multiproc
		if cte.Nb_CPU is None:
			Nb_CPU = mp.cpu_count()
		else:
			Nb_CPU = cte.Nb_CPU 
		if (self._args.origin=='gdr'):
		    PATH_ALTI_in,PATH_ALTI_ii,TAG_ALTI=alti.alti_paths(self._args.mission)
		else:
		    PATH_ALTI_in,PATH_ALTI_ii,TAG_ALTI=alti.alti_paths_cci(self._args.mission)
		for mm,yy in months_years:
			count = 0
			dt_mes0 = datetime.now() 
			filename = PATH_ALTI_in.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}')
			print('Will look for satellite files with this pattern:',filename)
			_filesave = alti.FORMAT_OUT_detalt.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}').replace('ALT',TAG_ALTI)
			filesave = os.path.join(cte.PATH_SAVE_detect,_filesave)
			if os.path.isfile(filesave):
				if (args.reprocess == True):
					os.remove(filesave)
					log.info("file removed for reprocessing %s"% filesave)
				else:
					log.info("file already exists and no reprocessing => skipping file \n %s"% filesave)
					continue
			file_list = sorted(glob.glob(filename))
			nfile=len(file_list) 
			if (len(PATH_ALTI_ii) > 10): 
			    filenamei = PATH_ALTI_ii.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}')
			    file_listi = sorted(glob.glob(filenamei))
			    nfilei=len(file_listi) 
			    if nfilei > nfile: 
			        nfile=nfilei
			        file_list=file_listi
			   
			#print('TEST:',nfile,filename)
			if nfile > 0: 
				if ismp:
					pool = mp.Pool(Nb_CPU)
					results = pool.starmap_async(get_storm_by_file, [(self._args.mission,self._args.origin,file_list[it],  yy,mm, alti.hs_thresh,alti.min_length,count0==count) for it in range(nfile) ]).get()
					pool.close()
					r_xr = xr.concat(results,dim='x').sortby('time')
				else:
					results = []
					for ifile in range(nfile): 
						#print('list:',file_list[ifile])
						#ds = read_WW3_HS_file(filepath,filename)
						_results,fulltrack,count = get_storm_by_file(self._args.mission,self._args.origin,file_list[ifile], yy,mm, alti.hs_thresh,alti.min_length,count0=count )
						if (len(_results) >0): 
							results.append(_results)
						log.info(' -- ifile = '+str(ifile)+' out of '+str(nfile))
					r_xr = xr.concat(results,dim='x').sortby('time')
				r_xr.to_netcdf(os.path.join(cte.PATH_SAVE_detect,filesave),unlimited_dims={'x':True})
				dt_mes = datetime.now() - dt_mes0
				log.info("---- Detection done for  "+_filesave+" time elapsed = "+str(dt_mes))
			
	def check_detected_exists(self,months_years,args):
		list_needed = [cte.FORMAT_OUT_detalt.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}') for mm,yy in months_years]
		list_exists = os.listdir(cte.PATH_SAVE_detect)
		need_in_exist = np.isin(list_needed,list_exists)
		
		if np.all(need_in_exist):
			log.info('All files have been processed through detection step: proceed with tracking')
		else:
			log.warn('Some files were not processed through detection step :')
			months_years_notdetect = months_years[np.where(need_in_exist==False)[0],:]
			for i,j in months_years_notdetect:
				log.warn(' - '+f'{i:02d}'+'/'+f'{j:04d}')
			log.warn('=> let proceed with detection step (automatically) before tracking (if this message repeats itself => stop the code, you may have entered an infinite loop !)')
			# The infinite loop is highly improbable in nominal conditions, however, if for any reasons the detection cannot be performed on the file, there might be a problem ... 
			self.run_detection(months_years_notdetect,args)
	
	def check_trackingtoberemoved(self,months_years,args):
		listfiles_in = []
		listfiles_out = []
		for mm,yy in months_years:
			_filename = cte.FORMAT_OUT_detect.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}')
			_filesave = cte.FORMAT_OUT_tracking.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}')
			filesave = os.path.join(cte.PATH_SAVE_tracking,_filesave)
			filesave2 = os.path.join(cte.PATH_SAVE_tracking,'__'+_filesave)
			
			if os.path.isfile(filesave):
				if (args.reprocess_tracking == True):
					os.remove(filesave)
					if os.path.isfile(filesave2):
						os.remove(filesave2)
					listfiles_out.append(_filesave)
					listfiles_in.append(_filename)
					log.info("file removed for reprocessing %s"% filesave)
				else:
					log.info("file already exists and no reprocessing => skipping file \n %s"% filesave)
					continue
			else:
				listfiles_out.append(_filesave)
				listfiles_in.append(_filename)
		return listfiles_in, listfiles_out
			
	def check_transifiles(self,months_years,args):
		mm0,yy0 = months_years[0]
		listfiles_out = []
		for mm,yy in months_years:
			_filesave = cte.FORMAT_OUT_tracking.replace('YYYY',f'{yy:04d}').replace('MM',f'{mm:02d}')
			filesave = os.path.join(cte.PATH_SAVE_tracking,_filesave)
			filesave0 = os.path.join(cte.PATH_SAVE_tracking,'__'+_filesave)
			if os.path.isfile(filesave):
				if (args.reprocess_tracking == True):
					os.remove(filesave)
					log.info("file removed for reprocessing %s"% filesave)
					if os.path.isfile(filesave0):
						listfiles_out.append(_filesave)
						if (mm == mm0) & (yy == yy0):
							shutil.copy(filesave0,filesave)
							log.info('Copy 1st file from individual tracking file')
					else:
						log.error('Individual tracking file %s does not exist => no transition can be run'% '__'+_filesave)
				else:
					if (mm == mm0) & (yy == yy0):
						log.info("first file already exists : don't change it")
						listfiles_out.append(_filesave)
					else:
						log.info("file already exists and no reprocessing => skipping file \n %s"%filesave)
						continue
			else:
				if os.path.isfile(filesave0):
					listfiles_out.append(_filesave)
					if (mm == mm0) & (yy == yy0):
						shutil.copy(filesave0,filesave)
						log.info('Copy 1st file from individual tracking file')
				else:
					log.error('Individual tracking file %s does not exist => no transition can be run'%'__'+_filesave)	
		return listfiles_out	
		
	def run_tracking(self,months_years,args):
		''' function that does the tracking inside a monthly file '''
		self.check_detected_exists(months_years,args)
		ismp = args.is_multiproc
		if cte.Nb_CPU is None:
			Nb_CPU = mp.cpu_count()
		else:
			Nb_CPU = cte.Nb_CPU
		
		listfiles_in, listfiles_out = self.check_trackingtoberemoved(months_years,args)
		if len(listfiles_in)<1:
			log.info('--- tracking: Nothing to be done ----')
		else:
			disttocoast = xr.open_dataarray(cte.filedist2coast)
			if ismp:
				log.info(' Tracking for individual files => Multiprocessing run')
				pool = mp.Pool(Nb_CPU)
				results = pool.starmap_async(track_for_1_file,[(fin,fout,cte.PATH_SAVE_detect, cte.PATH_SAVE_tracking, disttocoast,log, cte.threshold_dist) for fin,fout in zip(listfiles_in, listfiles_out)]).get()
				pool.close()
		 
				log.info('--- tracking: Done for all files individually (MP version) ----')
			else:
				log.info(' Tracking for individual files => For-loop run')
				for fin,fout in zip(listfiles_in, listfiles_out):
					track_for_1_file(fin,fout,cte.PATH_SAVE_detect, cte.PATH_SAVE_tracking, disttocoast,log, threshold_dist=cte.threshold_dist)
					log.info('--- tracking: Done for '+fout)
				log.info('--- tracking: Done for all files individually (For-loop version) ----')

	def run_transition_tracking(self,months_years,args):
		''' function that does the tracking between 2 monthly files '''
		listfiles_out = self.check_transifiles(months_years,args)
		if len(listfiles_out) != np.size(months_years,0):
			log.error('Check your internal file tracking before attempting to track transition between files')
		else:
			disttocoast = xr.open_dataarray(cte.filedist2coast)
			for if1,f1 in enumerate(listfiles_out[:-1]):
				f2 = listfiles_out[if1+1]
				track_for_1_transition(cte.PATH_SAVE_tracking,f1,f2,disttocoast,threshold_dist = cte.threshold_dist)
				log.info('Transition computed between %s and %s' % (f1,f2))
	
	def process(self,months_years,args,step=StepChoice.all_from_detect):
		'''
		Remember the different steps :
			detect_only = 0
			pairing_only = 1
			track_transition_only = 2
			all_from_tracking = 11 # steps  1 + 2
			all_from_detect = 111 # steps 0 + 1 + 2
			all_upto_transition = 110
		'''
		self.check_path(cte.PATH_SAVE_detect)
		self.check_path(cte.PATH_SAVE_tracking)
		
		if (step == StepChoice.detect_only) | (step == StepChoice.all_from_detect) | (step == StepChoice.all_upto_transition):
			log.info("|========================================== ")
			log.info("| Start of detection from (%s,%s) to (%s,%s)" % (months_years[0][0],months_years[0][1],months_years[-1][0],months_years[-1][1]))
			log.info("|========================================== ")
			self.run_detection(months_years,args)
			log.info("|============================================ ")
			log.info("|  => Detection from (%s,%s) to (%s,%s) Done ! " % (months_years[0][0],months_years[0][1],months_years[-1][0],months_years[-1][1]))
			log.info("|=========================================== ")
		if (step == StepChoice.tracking_only) | (step == StepChoice.all_from_tracking) | (step == StepChoice.all_upto_transition):
			log.info("|========================================== ")
			log.info("| Start of tracking inside individual files from (%s,%s) to (%s,%s)" % (months_years[0][0],months_years[0][1],months_years[-1][0],months_years[-1][1]))
			log.info("|========================================== ")
			self.run_tracking(months_years,args)
			log.info("|========================================== ")
			log.info("|  => Tracking inside individual files from (%s,%s) to (%s,%s) Done ! " % (months_years[0][0],months_years[0][1],months_years[-1][0],months_years[-1][1]))
			log.info("|========================================== ")
		if (step == StepChoice.track_transition_only) | (step == StepChoice.all_from_detect) | (step == StepChoice.all_from_tracking):
			if np.size(months_years,0) >1:
				log.info("|========================================== ")
				log.info("| Start of tracking transition between files from (%s,%s) to (%s,%s)" % (months_years[0][0],months_years[0][1],months_years[-1][0],months_years[-1][1]))
				log.info("|========================================== ")
				self.run_transition_tracking(months_years,args)
				log.info("|========================================== ")
				log.info("|  => Tracking transition between files from (%s,%s) to (%s,%s) Done ! " % (months_years[0][0],months_years[0][1],months_years[-1][0],months_years[-1][1]))
				log.info("|========================================== ")
			else:
				if (step == StepChoice.all_from_detect):
					log.error('Transition between files can not be run if only one month is to be treated, consider running step 110 instead of 111')
				if (step == StepChoice.all_from_tracking):
					log.error('Transition between files can not be run if only one month is to be treated, consider running step 1 instead of 11')   
				else:
					log.error('Transition between files can not be run if only one month is to be treated')
		 
	def init_option(self):
		parser = StormParser(prog="detect_track_storms")#,
							# usage="%(prog)s [-yyear] [-fyfinal_year] [-mmonth] [-fmfinal_month] [-s'0|1']",
							# description="%(prog)s v1.1")
		parser.add_argument("-n","--mission",dest="mission",help="Chose the satellite mission for processing",action="store", default=None, type=str)
		
		parser.add_argument("-o","--origin",dest="origin",help="Chose the data source",action="store", default='gdr', type=str)
		parser.add_argument("-y","--year",dest="year",help="Chose the year for processing",action="store", default=None, type=int)
		parser.add_argument("-m","--month",dest="month",help="Chose the month for processing",action="store", default=None, type=int)
		parser.add_argument("-Y","--final_year",dest="final_year",help="Chose the final year to take into account",action="store", default=None, type=int)
		parser.add_argument("-M","--final_month",dest="final_month",help="Chose the final month to take into account",action="store", default=None, type=int)
		parser.add_argument("-s", "--step", dest="step",help="Choose the step for applying the calculation process. 0: detect_only | 1: file_internal_track_only | 2: transition_track_only | 111 : all_steps | 110: detect and file internal track steps (no transition between files track) | 11 : all_tracking_steps",action="store", default=0, type=int, choices=[i.value for i in StepChoice.__iter__()])
		parser.add_argument("-r", "--reprocess", dest="reprocess",
							help="Force a reprocessing for the detection step : existing files will be deleted and re computed",
							action="store_true", default=False)
		parser.add_argument("-R","--reprocess_tracking", dest="reprocess_tracking",
							help="Force a reprocessing for the tracking steps : existing files will be deleted and re computed",
							action="store_true", default=False)
		parser.add_argument("--multiprocessing", dest="is_multiproc",
							help="Use multiprocessing to run the code",
							action="store_true", default=False)		   
		return parser.parse_args()
		
	def init_dates(self):
		final_month = 12
		final_year = None
		month = 1
		year = 1993
		if self._args.year is None:
			log.warn('--year parameter is not set => will run for year=1993')
		else:
			year = self._args.year
		if self._args.final_year is None:
			log.warn('--final_year parameter is not set => will run for final_year='+str(year))
			final_year = year
		else:
			final_year = self._args.final_year
		if self._args.month is None:
			if self._args.final_month is None:
				log.warn('Neither --month nor --final_month parameters are set => will run for month = 1 and final_month = 12')
			else:
				log.warn('--month parameter is not set => will run for month = 1')
				final_month = self._args.final_month
		else:
			month = self._args.month
			if self._args.final_month is None:
				log.warn('--final_month parameter is not set => will run for final_month = '+str(month))
				final_month = month
			else:
				final_month = self._args.final_month
		
		return year, final_year, month, final_month

########################################"		
	def main(self):
		# read the input parameters
		self._args = self.init_option()
		year, final_year, month, final_month= self.init_dates() # needs to have self._args (always after the previous line)
		step = self._args.step

		log.info('|##############################################################################')
		log.info('| Start of Processing '+StepChoice(step)._name_+' from '+f'{month:02d}'+'/'+f'{year:04d}'+' to ' + f'{final_month:02d}'+'/'+f'{final_year:04d}')
		log.info('|##############################################################################')
				
		months_years = self.get_all_months(year, final_year, month, final_month)
		log.info('| '+str(np.size(months_years,0))+' time steps to be processed ')
		self.process(months_years, self._args, step=step)
		
	
def main():
	detect_tracking = StormDetectionTracking()
	detect_tracking.main()

if __name__ == "__main__":
	main()

