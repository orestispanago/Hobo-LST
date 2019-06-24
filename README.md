# Hobo-LST
Hobo-MODIS LST comparison

**LST_time_skip_nans_hobo.py:** 

Reads LST value and time from MODIS Terra and Aqua .hdf files (day and night observations)

Reads Hobo data and compares timeseries

Calculates regression parameters

Applies correction factor to LST data 

Plots SUHII with and without correction factor


**LST_time_skip_nans_hobo_subprocess.py:** 

Attempt to speed up code using subprocess module.

(PC freezes sometimes)