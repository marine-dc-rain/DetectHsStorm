# DetectHsStorms
This is a python code for detecting and tracking "storms" of Significant wave height (Hs) in models.

N.B: More infos on the code can be found in **Readme.pdf**


## Environment requirements
The following packages are needed:
- matplotlib
- pandas
- numpy
- scipy
- xarray
- netcdf4
- h5netcdf

To use the environment in jupyter :  
`<whatever_snake> install ipykernel jupyter`  
`python -m ipykernel install --user --name=<ENV_name> --display-name=<ENV_name_to_be_displayed>`

## To run this code 
1) Activate your python environment
2) go to 'src' folder
3) Update the detection_code/params_detect.py file with your configuration
4) launch the computation by typing (here is an example for applying detection only to January and February 2023):  
  `python detect_storms_in_model.py -s 0 -y 2023 -Y 2023 –m 1 –M 2`


## Options to run the code
```
python detect_storms_in_model.py -h

usage: detect_track_storms [-h] [-y YEAR] [-m MONTH] [-Y FINAL_YEAR] [-M FINAL_MONTH] [-s {0,1,2,11,111,110}] [-r] [-R]
[--multiprocessing]
options:
-h, --help
show this help message and exit
-y YEAR, --year YEAR Chose the year for processing
-m MONTH, --month MONTH
Chose the month for processing
-Y FINAL_YEAR, --final_year FINAL_YEAR
Chose the final year to take into account
-M FINAL_MONTH, --final_month FINAL_MONTH
Chose the final month to take into account
-s {0,1,2,11,111,110}, --step {0,1,2,11,111,110}
Choose the step for applying the calculation process. 0: detect_only | 1: file_internal_track_only | 2:
transition_track_only | 111 : all_steps | 110: detect and file internal track steps (no transition between files
track) | 11 : all_tracking_steps
-r, --reprocess Force a reprocessing for the detection step : existing files will be deleted and re computed
-R, --reprocess_tracking
Force a reprocessing for the tracking steps : existing files will be deleted and re computed
--multiprocessing Use multiprocessing to run the code
```
