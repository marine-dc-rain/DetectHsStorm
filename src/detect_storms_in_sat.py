import sys
import os
import glob as glob
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

# from detection_code.params_altimeters import get_storm_by_file
from detection_code.storms_functions_detect import get_storms_track_from_sat_by_file

logbook.StreamHandler(sys.stdout).push_application()
log = logbook.Logger('StormAnalysis ')


class StepChoice(IntEnum):
    detect_only = 0
    concat = 1
    # tracking_only = 1
    # track_transition_only = 2
    # all_from_tracking = 11  # steps  1 + 2
    # all_from_detect = 111  # steps 0 + 1 + 2
    # all_upto_transition = 110


class StormParser(ArgumentParser):
    def error(self, message):
        log.error(message)
        self.print_help()
        exit(2)


class SatStormDetection:
    def check_path(self, dir_path):
        if not os.path.isdir(dir_path):
            if os.path.exists(dir_path):
                # exists but is not a directory
                log.error('%s already exists but is not a directory !' % dir_path)
            else:
                os.makedirs(dir_path)
                log.info('%s : Directory created successfully' % dir_path)
        else:
            log.info("%s : Directory already exists!" % dir_path)

    def get_all_months(self, year, final_year, month, final_month):
        day_start = pd.Timestamp(year, month, 1)
        day_end = pd.Timestamp(final_year, (final_month), 1)
        A = pd.date_range(day_start, day_end, freq='MS')
        return np.vstack((A.month, A.year)).T

    def get_filesave_name(self, mm, yy, TAG_ALTI):
        _filesave = (
            alti.FORMAT_OUT_detalt.replace('YYYY', f'{yy:04d}').replace('MM', f'{mm:02d}').replace('ALT', TAG_ALTI)
        )
        filepathsave = os.path.join(alti.PATH_SAVE_det_sat, _filesave)
        log.info("file to save %s" % filepathsave)
        if os.path.isfile(filepathsave):
            if self._args.reprocess == True:
                os.remove(filepathsave)
                log.info("file removed for reprocessing %s" % filepathsave)
            else:
                log.info("file already exists and no reprocessing => skipping file \n %s" % filepathsave)
                return 0, None
        return filepathsave, _filesave

    def get_list_file_by_month(self, mm, yy, PATH_ALTI_in, PATH_ALTI_ii):
        filename = PATH_ALTI_in.replace('YYYY', f'{yy:04d}').replace('MM', f'{mm:02d}')

        file_list = sorted(glob.glob(filename))
        nfile = len(file_list)
        print('Will look for satellite files with this pattern:', filename, nfile)
        if len(PATH_ALTI_ii) > 10:
            filenamei = PATH_ALTI_ii.replace('YYYY', f'{yy:04d}').replace('MM', f'{mm:02d}')
            file_listi = sorted(glob.glob(filenamei))
            nfilei = len(file_listi)
            if nfilei >= nfile:
                nfile = nfilei
                file_list = file_listi
        return file_list, nfile

    def get_all_list_files(self, PATH_ALTI_in):
        filename = PATH_ALTI_in.replace('YYYY', '*').replace('MM*', '')
        file_list = sorted(glob.glob(filename))
        nfile = len(file_list)
        return file_list, nfile

    def run_detection(self, months_years):
        ismp = self._args.is_multiproc
        ###
        if cte.Nb_CPU is None:
            Nb_CPU = mp.cpu_count()
        else:
            Nb_CPU = cte.Nb_CPU
        ###
        if self._args.origin == 'gdr':
            PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI, addvarlist = alti.alti_paths_GDR(self._args.mission)
        if self._args.origin == 'cci':
            PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI, addvarlist = alti.alti_paths_cci(self._args.mission)
        if self._args.origin == 'l2p':
            PATH_ALTI_in, PATH_ALTI_ii, TAG_ALTI, addvarlist = alti.alti_paths_cci(
                self._args.mission, origin_type='l2p', version=self._args.version
            )

        for mm, yy in months_years:
            count = 0
            dt_mes0 = datetime.now()
            filepathsave, _filesave = self.get_filesave_name(mm, yy, TAG_ALTI)
            if filepathsave == 0:
                continue
            file_list, nfile = self.get_list_file_by_month(mm, yy, PATH_ALTI_in, PATH_ALTI_ii)
            # print('TEST:',nfile,filename)
            if nfile == 0:
                log.info(f"---- No input files found for {TAG_ALTI} on month {mm}/{yy}")
                continue
            if ismp:
                pool = mp.Pool(Nb_CPU)
                results0 = pool.starmap_async(
                    get_storms_track_from_sat_by_file,
                    [
                        (
                            self._args.mission,
                            self._args.origin,
                            file_list[it],
                            yy,
                            mm,
                            alti.hs_thresh,
                            alti.min_length,
                            count,
                            addvarlist,
                            False,
                            False,
                            alti.hs_thresh_min,
                        )
                        for it in range(nfile)
                    ],
                ).get()
                pool.close()
                results = [r[0] for r in results0 if r[0] is not None]
                # max_segm = [r.segments.values.max for r in results]
                # len_file = [r.sizes['x'] for r in results]

                if len(results) > 0:
                    r_xr = xr.concat(results, dim='x').sortby('time')
                else:
                    log.info(f"No results (above thresh) for {TAG_ALTI} on month {mm}/{yy}")
            else:
                results = []
                for ifile, file0 in enumerate(file_list):
                    _results, _, count = get_storms_track_from_sat_by_file(
                        self._args.mission,
                        self._args.origin,
                        file0,
                        yy,
                        mm,
                        alti.hs_thresh,
                        alti.min_length,
                        count0=count,
                        addvarlist=addvarlist,
                        hs_thresh_min=alti.hs_thresh_min,
                    )
                    if _results is not None:
                        results.append(_results)
                    log.info(' -- ifile = ' + str(ifile) + ' out of ' + str(nfile))
                if len(results) > 0:
                    r_xr = xr.concat(results, dim='x').sortby('time')
            if len(results) > 0:
                r_xr.to_netcdf(filepathsave, unlimited_dims={'x': True})
            dt_mes = datetime.now() - dt_mes0
            log.info("---- Detection done for  " + _filesave + " time elapsed = " + str(dt_mes))

    def concat_month_files(self):
        if self._args.origin == 'gdr':
            _, _, TAG_ALTI = alti.alti_paths_GDR(self._args.mission)
        if self._args.origin == 'cci':
            _, _, TAG_ALTI = alti.alti_paths_cci(self._args.mission)
        if self._args.origin == 'l2p':
            _, _, TAG_ALTI = alti.alti_paths_cci(self._args.mission, origin_type='l2p', version=self._args.version)
        _filesave = (
            alti.FORMAT_OUT_detalt.replace('YYYY', '*').replace('MM', '*').replace('**', '*').replace('ALT', TAG_ALTI)
        )
        filepathsaved = os.path.join(alti.PATH_SAVE_det_sat, _filesave)
        list_files = glob.glob(filepathsaved)

        list_files.sort()
        ds = xr.open_mfdataset(list_files, combine='nested', concat_dim='x')
        ds.time.values = np.array(ds.time, dtype='timedelta64[s]') + pd.Timestamp(str(ds.reference_time.values[()]))
        ds = ds.drop_vars('reference_time')
        segs = ds.segments.values
        # -100 is for the last one
        ind = np.nonzero((segs[0:-1] == -100) & (segs[1:] != -100))[0]
        ind = np.concatenate([np.array([-1]), ind, np.array([len(segs) - 1])])
        size_rep = np.diff(ind)
        new_segs = np.repeat(np.arange(len(ind) - 1), size_rep)
        ds = ds.assign(segments=ds.segments.copy(data=new_segs))
        ds = ds.assign(sat=(['x'], np.full(ds.sizes['x'], TAG_ALTI)))

        # ----
        _filename = (
            alti.FORMAT_OUT_detalt_summary.replace('T1', str(int(ds.yearmonth.values[0])))
            .replace('T2', str(int(ds.yearmonth.values[-1])))
            .replace('ALT', TAG_ALTI)
            .replace('TYPE', 'all')
        )
        filesaveall = os.path.join(alti.PATH_SAVE_summary_sat, _filename)
        ds.to_netcdf(filesaveall)
        log.info(f'Save concatenated file with entire tracks for {filesaveall}')

        # ----
        def sat_by_storm(ds):
            ind = np.argmax(ds.swh_1hz.values)
            return ds.isel(x=ind)

        ds_sat_bystorm = ds.groupby('segments').map(sat_by_storm)
        ds_sat_bystorm = ds_sat_bystorm.rename_vars({'segments': 'numStorm'}).swap_dims({"segments": "x"})
        _filename = (
            alti.FORMAT_OUT_detalt_summary.replace('T1', str(int(ds.yearmonth.values[0])))
            .replace('T2', str(int(ds.yearmonth.values[-1])))
            .replace('ALT', TAG_ALTI)
            .replace('TYPE', 'max')
        )
        filesaveall = os.path.join(alti.PATH_SAVE_summary_sat, _filename)
        ds_sat_bystorm.to_netcdf(filesaveall)
        log.info(f'Save concatenated file with max for segments tracks for {filesaveall}')

    def process(self, months_years, step=StepChoice.detect_only):
        '''
        For sat process only the detection step
        '''
        self.check_path(cte.PATH_SAVE_detect)
        self.check_path(cte.PATH_SAVE_tracking)
        if step == 0:
            log.info("|========================================== ")
            log.info(
                "| Start of detection from (%s,%s) to (%s,%s)"
                % (months_years[0][0], months_years[0][1], months_years[-1][0], months_years[-1][1])
            )
            log.info("|========================================== ")
            self.run_detection(months_years)
            log.info("|============================================ ")
            log.info(
                "|  => Detection from (%s,%s) to (%s,%s) Done ! "
                % (months_years[0][0], months_years[0][1], months_years[-1][0], months_years[-1][1])
            )
            log.info("|=========================================== ")
        elif step == 1:
            log.info("|========================================== ")
            log.info(
                "| Start of concatenation from (%s,%s) to (%s,%s)"
                % (months_years[0][0], months_years[0][1], months_years[-1][0], months_years[-1][1])
            )
            log.info("|========================================== ")
            self.concat_month_files()
            log.info("|============================================ ")
            log.info(
                "|  => Concatenation from (%s,%s) to (%s,%s) Done ! "
                % (months_years[0][0], months_years[0][1], months_years[-1][0], months_years[-1][1])
            )
            log.info("|=========================================== ")
        # if np.size(months_years, 0) > 1:
        #     log.info("|========================================== ")
        #     log.info(
        #         "| Start of tracking transition between files from (%s,%s) to (%s,%s)"
        #         % (months_years[0][0], months_years[0][1], months_years[-1][0], months_years[-1][1])
        #     )
        #     log.info("|========================================== ")
        #     self.run_transition_tracking(months_years, args)
        #     log.info("|========================================== ")
        #     log.info(
        #         "|  => Tracking transition between files from (%s,%s) to (%s,%s) Done ! "
        #         % (months_years[0][0], months_years[0][1], months_years[-1][0], months_years[-1][1])
        #     )
        #     log.info("|========================================== ")
        # else:
        #     if step == StepChoice.all_from_detect:
        #         log.error(
        #             'Transition between files can not be run if only one month is to be treated, consider running step 110 instead of 111'
        #         )
        #     if step == StepChoice.all_from_tracking:
        #         log.error(
        #             'Transition between files can not be run if only one month is to be treated, consider running step 1 instead of 11'
        #         )
        #     else:
        #         log.error('Transition between files can not be run if only one month is to be treated')

    def init_option(self):
        parser = StormParser(prog="detect_storms_in_sat")  # ,
        # usage="%(prog)s [-n mission] [-o origin['gdr'|'cci'|'l2p']][-yyear] [-fyfinal_year] [-mmonth] [-fmfinal_month] [-s'0|1']",
        # description="%(prog)s v1.1")
        parser.add_argument(
            "-n",
            "--mission",
            dest="imission",
            help="Chose the satellite mission for processing",
            action="store",
            default=None,
            type=int,
        )

        parser.add_argument(
            "-o", "--origin", dest="origin", help="Chose the data source", action="store", default='gdr', type=str
        )
        parser.add_argument(
            "-s",
            "--step",
            dest="step",
            help="Choose the step for applying the calculation process. 0: detect_only | 1: concatenate over months ",
            action="store",
            default=0,
            type=int,
            choices=[i.value for i in StepChoice.__iter__()],
        )
        parser.add_argument(
            "-y", "--year", dest="year", help="Chose the year for processing", action="store", default=None, type=int
        )
        parser.add_argument(
            "-m", "--month", dest="month", help="Chose the month for processing", action="store", default=None, type=int
        )
        parser.add_argument(
            "-Y",
            "--final_year",
            dest="final_year",
            help="Chose the final year to take into account",
            action="store",
            default=None,
            type=int,
        )
        parser.add_argument(
            "-M",
            "--final_month",
            dest="final_month",
            help="Chose the final month to take into account",
            action="store",
            default=None,
            type=int,
        )
        parser.add_argument(
            "-r",
            "--reprocess",
            dest="reprocess",
            help="Force a reprocessing for the detection step : existing files will be deleted and re computed",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "--multiprocessing",
            dest="is_multiproc",
            help="Use multiprocessing to run the code",
            action="store_true",
            default=False,
        )
        parser.add_argument(
            "-v",
            dest="version",
            help="CCI version to be used",
            action="store",
            default='v5',
            type=str,
        )
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
            log.warn('--final_year parameter is not set => will run for final_year=' + str(year))
            final_year = year
        else:
            final_year = self._args.final_year
        if self._args.month is None:
            if self._args.final_month is None:
                log.warn(
                    'Neither --month nor --final_month parameters are set => will run for month = 1 and final_month = 12'
                )
            else:
                log.warn('--month parameter is not set => will run for month = 1')
                final_month = self._args.final_month
        else:
            month = self._args.month
            if self._args.final_month is None:
                log.warn('--final_month parameter is not set => will run for final_month = ' + str(month))
                final_month = month
            else:
                final_month = self._args.final_month

        return year, final_year, month, final_month

    ########################################"
    def main(self):
        # read the input parameters
        self._args = self.init_option()
        sats = [
            'cfosat',
            'cryosat2',  # 1
            'envisat',
            'ers1',  # 3
            'ers2',  # 4
            'gfo',
            'jason1',  # 6
            'jason2',  # 7
            'jason3',  # 8
            'saral',
            'sentinel3a',  # 10
            'sentinel3b',  # 11
            'sentinel6a',  # 12
            'swot',
            'tp-topex',  # 14
            'tp-poseidon',  # 15
        ]
        self._args.mission = sats[self._args.imission]
        year, final_year, month, final_month = (
            self.init_dates()
        )  # needs to have self._args (always after the previous line)

        log.info('|##############################################################################')
        log.info(
            '| Start of Processing Sat detection from '
            + f'{month:02d}'
            + '/'
            + f'{year:04d}'
            + ' to '
            + f'{final_month:02d}'
            + '/'
            + f'{final_year:04d}'
        )
        log.info('|##############################################################################')

        months_years = self.get_all_months(year, final_year, month, final_month)
        log.info('| ' + str(np.size(months_years, 0)) + ' time steps to be processed ')
        self.process(months_years, step=self._args.step)


def main():
    detect_in_sat = SatStormDetection()
    detect_in_sat.main()


if __name__ == "__main__":
    main()
