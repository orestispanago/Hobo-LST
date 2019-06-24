import os
import glob
import subprocess
import time
from datetime import datetime
import pandas as pd


start = time.time()

hdfiles = glob.glob(os.getcwd() + '/raw/*/' + '*.hdf')
when='Day'
lat = 38.291969
lon = 21.788156

class Hobo:
    stations = []
    def __init__(self, codename, alias, lat, lon):
        self.codename = codename
        self.alias = alias
        self.lat = lat
        self.lon = lon
        Hobo.stations.append(self)
        
        
def get_processes(lat,lon,quantity='lst',when='Day'):
    procs = []
    for filename in hdfiles:
        if quantity is 'time':
            subdataset = f"HDF4_EOS:EOS_GRID:{filename}:MODIS_Grid_Daily_1km_LST:{when}_view_time"
        elif quantity is 'lst':
            subdataset = f"HDF4_EOS:EOS_GRID:{filename}:MODIS_Grid_Daily_1km_LST:LST_{when}_1km"
        proc = subprocess.Popen(f'gdallocationinfo -valonly {subdataset} -wgs84 {lon} {lat}',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        procs.append(proc)
    return procs

def get_process_output(proc_list):
    proc_out = []
    for proc in proc_list:
        out, err = proc.communicate()
        out = out.decode().strip()
        proc_out.append(out)
    return proc_out

def convert_lst(a):
    """ Converts string to LST in Celsius"""
    if int(a) is 0:
        return None
    return int(a)*0.02-273.14

def convert_time(stime):
    if int(stime) is 255:
        return None,None
    desc_stime = int(stime) * 0.1
    utime = float(desc_stime) - 21.7 / 15  # Convert local solar time to UTC
    uhour = str(int(utime)).zfill(2)
    umin = str(round((utime % 1) * 60)).zfill(2)
    return uhour, umin

def to_datetime(soltime):
    datetimes = []
    for filename, j in zip(hdfiles,soltime):
        year_day = filename.split('.')[1]
        hour,minute = convert_time(j)
        if minute or hour is not None:
            dtime = datetime.strptime(year_day + hour + minute, 'A%Y%j%H%M')
        else:
            dtime=None
        datetimes.append(dtime)
    return datetimes

df = pd.read_csv('coords.txt')

for i in range(len(df)):
    Hobo(*df.loc[i].tolist())


dflist = []
for station in Hobo.stations:
    df1 = pd.DataFrame(columns=['time','lst','alias'])
    lst_list = []
    date_list = []
    for dn in ['Day','Night']:
        lst_procs = get_processes(station.lat,station.lon,quantity='lst',when=dn)
        stime_procs = get_processes(station.lat,station.lon,quantity='time',when=dn)
        lst = get_process_output(lst_procs)
        soltimelist = get_process_output(stime_procs)
    
        lst_c = [convert_lst(i) for i in lst]
        date_time = to_datetime(soltimelist)
        lst_list.extend(lst_c)    
        date_list.extend(date_time)   
        
    df1['time'] = date_list
    df1['lst'] = lst_list
    df1['alias'] = station.alias
    dflist.append(df1)
    
lst_df = pd.concat(dflist)
lst_df = lst_df.pivot_table('lst', 'time', 'alias')
    
    
    
print(time.time()-start)

