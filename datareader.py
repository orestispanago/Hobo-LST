import os
import glob
import pandas as pd
import datetime


class Hobo:
    stations = []

    def __init__(self, codename, alias, lat, lon):
        self.codename = codename
        self.alias = alias
        self.lat = lat
        self.lon = lon
        Hobo.stations.append(self)

    def read_each(self, datadir):
        """ Reads .csv file for a Hobo object """
        csvlist = glob.glob(f'{os.getcwd()}/{datadir}/*{self.codename}*.csv')
        each = []
        for fname in csvlist:
            hob = pd.read_csv(fname, skiprows=2, usecols=[1, 2],
                              names=['time', self.alias + 'T'],
                              index_col='time', parse_dates=True)
            hob = hob.tz_localize('UTC')
            each.append(hob)
        each_df = pd.concat(each, axis=0)
        each_df = each_df.dropna()
        return each_df

    def read_all(datadir):
        """ Loads dataset from all Hobo objects """
        df_list = []
        for station in Hobo.stations:
            each = Hobo.read_each(station, datadir)
            df_list.append(each)
        hobo_df = pd.concat(df_list, axis=0, sort=False)
        hobo_df = hobo_df.resample('30min').mean()
        return hobo_df


def init_hobo():
    coords = pd.read_csv('stations.txt')
    for row in range(len(coords)):
        Hobo(*coords.loc[row].tolist())


class Modis:

    def descale_lst(vals):
        descvals = []
        for i in vals:
            if i != '0':
                descvals.append(int(i) * 0.02 - 273.15)
            else:
                descvals.append(None)
        return descvals

    def desc_conv_time(filename, vals):
        """Descales time and converts from local solar to utc """
        year_day = filename.split('.')[1]
        utc = []
        for i in vals:
            if i != "255":
                i = int(i) * 0.1
                utime = float(i) - 21.7 / 15  # Convert local solar time to UTC
                uhour = str(int(utime)).zfill(2)
                umin = str(round((utime % 1) * 60)).zfill(2)
                dtime = datetime.datetime.strptime(year_day + uhour + umin, 'A%Y%j%H%M')
                utc.append(dtime)
            else:
                utc.append(None)
        return utc

    def read(datadir):
        hdfiles = glob.glob(os.getcwd() + datadir)

        df_lst = pd.DataFrame(columns=['alias', 'time', 'lst'])
        for when in ['Day', 'Night']:
            for filename in hdfiles:
                lst_subdataset = f"HDF4_EOS:EOS_GRID:{filename}:MODIS_Grid_Daily_1km_LST:LST_{when}_1km"
                valstring = os.popen(f'gdallocationinfo -valonly {lst_subdataset} -wgs84 < coords.txt').read()
                vals = valstring.split('\n')[:-1]
                # lst = [None if i=='0' else int(i)*0.02 -273.15 for i in vals] # Descale and conv K to C
                lst = Modis.descale_lst(vals)

                time_subdataset = f"HDF4_EOS:EOS_GRID:{filename}:MODIS_Grid_Daily_1km_LST:{when}_view_time"
                valstring = os.popen(f'gdallocationinfo -valonly {time_subdataset} -wgs84 < coords.txt').read()
                vals = valstring.split('\n')[:-1]
                utime = Modis.desc_conv_time(filename, vals)

                for station, tstamp, val in zip(Hobo.stations, utime, lst):
                    df_lst = df_lst.append(
                        {'alias': station.alias, 'time': tstamp, 'lst': val},
                        ignore_index=True)
        df_lst = df_lst.pivot_table('lst', 'time', 'alias')
        df_lst = df_lst.tz_localize('UTC')
        return df_lst

def load_dataset():
    """ Loads hobo and MODIS dataframes to one """
    hobo = Hobo.read_all('raw/Hobo-Apr-Nov')
    lst = Modis.read('/raw/*/*.hdf')
    lst = lst.resample('30min').mean()
    large = pd.concat([lst, hobo], sort=False)
    return large

init_hobo()

dataset = load_dataset()