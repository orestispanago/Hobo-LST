# Hobo-LST
Hobo-MODIS LST comparison

Faster and more stable version of .hdf reader functions.

**datareader.py:** 

* Contains Hobo and Modis classes, used to read data

**main.py**

* Reads LST and time from MODIS Terra and Aqua .hdf files (day and night observations)

* Reads Hobo data from .csv files

* Calculates regression parameters

* Applies correction factor to LST data 

* Selects day/night heat island indices

* Plots SUHII with and without correction factor
