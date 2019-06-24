import os
import glob
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import statsmodels.formula.api as sm
from statsmodels.api import add_constant

import time

start = time.time()


class Hobo:
    stations = []

    def __init__(self, codename, alias, lat, lon):
        self.codename = codename
        self.alias = alias
        self.lat = lat
        self.lon = lon
        Hobo.stations.append(self)


def pixel_utc(filename, lat, lon, when='Day'):
    year_day = filename.split('.')[1]
    time_subdataset = f"HDF4_EOS:EOS_GRID:{filename}:MODIS_Grid_Daily_1km_LST:{when}_view_time"
    stime = os.popen(
        f'gdallocationinfo -valonly {time_subdataset} -wgs84 {lon} {lat}').read()
    if stime == '255\n':
        return None
    desc_stime = int(stime) * 0.1
    utime = float(desc_stime) - 21.7 / 15  # Convert local solar time to UTC
    uhour = str(int(utime)).zfill(2)
    umin = str(round((utime % 1) * 60)).zfill(2)
    dtime = datetime.strptime(year_day + uhour + umin, 'A%Y%j%H%M')
    return dtime


def pixel_lst(filename, lat, lon, when='Day'):
    LST_subdataset = f"HDF4_EOS:EOS_GRID:{filename}:MODIS_Grid_Daily_1km_LST:LST_{when}_1km"
    LST = os.popen(
        f'gdallocationinfo -valonly {LST_subdataset} -wgs84 {lon} {lat}').read()
    if LST == '0\n':
        return None
    return int(LST) * 0.02 - 273.15


def stations_time_lst():
    hdfiles = glob.glob(os.getcwd() + '/raw/*/' + '*.hdf')
    df_lst = pd.DataFrame(columns=['alias', 'time', 'lst'])
    for filename in hdfiles:
        for station in Hobo.stations:
            for dn in ['Day', 'Night']:
                tstamp = pixel_utc(filename, station.lat, station.lon, when=dn)
                val = pixel_lst(filename, station.lat, station.lon, when=dn)
                df_lst = df_lst.append(
                    {'alias': station.alias, 'time': tstamp, 'lst': val},
                    ignore_index=True)
    df_lst = df_lst.pivot_table('lst', 'time', 'alias')
    return df_lst


def get_hobo():
    df_list = []
    for station in Hobo.stations:
        hobos = glob.glob(
            os.getcwd() + f'/raw/Hobo-Apr-Nov/*{station.codename}*.csv')
        each = []
        for fname in hobos:
            hob = pd.read_csv(fname, skiprows=2, usecols=[1, 2],
                              names=['time', station.alias + 'T'],
                              index_col='time', parse_dates=True)
            each.append(hob)
        each_df = pd.concat(each, axis=0)
        each_df = each_df.dropna()
        df_list.append(each_df)
    hobo_df = pd.concat(df_list, axis=0, sort=False)
    hobo_df = hobo_df.resample('30min').mean()
    return hobo_df


def plot_hobo_modis(df, folder=None):
    for station in Hobo.stations:
        plt.figure(figsize=(14, 4))
        plt.plot(df[station.alias + 'T'])
        plt.plot(df[station.alias])
        plt.title(station.alias)
        plt.legend(('Hobo', 'MODIS'))
        plt.ylabel('Temperature $(\degree$C)')
        plt.xlabel('Datetime - UTC')
        if folder is not None:
            name = station.alias + '.png'
            plt.savefig(make_dir(folder) + name)
        plt.show()


def plot_scatters(df, folder=None, name=None):
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True,
                             figsize=(14, 7))
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False,
                    right=False)
    for count, station in enumerate(Hobo.stations):
        couple = df[[station.alias, f"{station.alias}T"]]
        a = couple.resample('30min').mean()
        a = a.dropna()
        test = a[station.alias].values  # Sat
        ref = a[f"{station.alias}T"].values  # Ground
        slope, intercept, r_value, p_value, std_err = stats.linregress(ref,
                                                                       test)
        g = sns.regplot(x=ref, y=test, ax=axes[count // 4][count % 4],
                        scatter_kws={'s': 4},
                        line_kws={'label': "$y={0:.1f}x+{1:.1f}$".format(slope,
                                                                         intercept)})
        g.legend()
        g.set(title=station.alias)
    fig.suptitle('Temperature ( $\degree$C)')
    plt.xlabel('HOBO')
    plt.ylabel('LST')
    if folder is not None:
        plt.savefig(make_dir(folder) + name)
    plt.show()


def reg_stats(df):
    reg_df = pd.DataFrame(
        columns=['slope', 'intercept', 'slopeSE', 'interceptSE', 'slopeP',
                 'interceptP', 'R2', 'RMSE', 'MBE', 'count'], )
    for station in Hobo.stations:
        couple = df[[station.alias, f"{station.alias}T"]]
        a = couple.resample('30min').mean()
        a = a.dropna()
        test = a[station.alias].values  # Sat
        ref = a[f"{station.alias}T"].values  # Ground

        X = add_constant(ref)  # include constant (intercept) in ols model
        mod = sm.OLS(test, X)
        results = mod.fit()

        inter, slope = results.params
        inter_stderr, slope_stderr = results.bse
        inter_p, slope_p = results.pvalues
        rsquared = results.rsquared
        num_values = results.nobs
        mbe = results.resid.sum() / results.nobs  # Mean Bias Error
        rmse = np.sqrt(results.ssr / results.nobs)  # Root Mean Square Error

        regstats = [slope, inter, slope_stderr, inter_stderr,
                    slope_p, inter_p, rsquared, rmse, mbe, num_values]
        reg_df.loc[station.alias] = regstats
    return reg_df


def make_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return f"{os.getcwd()}/{dirname}/"


def uhi_suhi(df):
    colnames = [other.alias for other in others]
    uhi = pd.DataFrame(columns=colnames)
    uhi.name = 'UHII'
    suhi = pd.DataFrame(columns=colnames)
    suhi.name = 'SUHII'
    for other in others:
        uhi[other.alias] = df[f"{other.alias}T"] - df[
            f"{ref_hobo.alias}T"]
        suhi[other.alias] = df[other.alias] - df[ref_hobo.alias]
    return uhi, suhi


def plot_hii(df, title=None, folder=None):
    """Plots heat island index (uhii or suhii)"""
    plt.figure(figsize=(14, 4))
    for i in others:
        plt.plot(df[i.alias])
        plt.legend()
    plt.title(title)
    if folder is not None:
        plt.savefig(make_dir(folder) + title + '.png')
    plt.show()


def cor_modis(df):
    """ Applies correction factor to MODIS data based on ground stations"""
    for i in regresults.index:
        slope = regresults.loc[i]['slope']
        intercept = regresults.loc[i]['intercept']
        df[i] = large[i] / slope - intercept
    return df


coords = pd.read_csv('coords.txt')

for row in range(len(coords)):
    Hobo(*coords.loc[row].tolist())

ref_hobo = Hobo.stations[0]
others = [station for station in Hobo.stations[1:]]

lst = stations_time_lst()
lst = lst.resample('30min').mean()
hobo = get_hobo()
large = pd.concat([lst, hobo], sort=False)

regresults = reg_stats(large)
regresults.to_excel(make_dir('No-Cor/Reg') + 'results.xlsx')
plot_scatters(large, folder='No-Cor/Reg', name='hobo_modis_scatter.png')

largec = cor_modis(large.copy())
regresultsc = reg_stats(largec)
regresultsc.to_excel(make_dir('Cor-Factor/Reg') + 'results.xlsx')
plot_scatters(largec, folder='Cor-Factor/Reg', name='hobo_modis_scatter.png')

daily = large.resample('D').mean()
plot_hobo_modis(daily, folder='No-Cor/Ts/Daily')

dailyc = largec.resample('D').mean()
plot_hobo_modis(dailyc, folder='Cor-Factor/Ts/Daily')

uhii, suhii = uhi_suhi(large)
uhiiw = uhii.resample('W').mean()
suhiiw = suhii.resample('W').mean()
plot_hii(uhiiw, title='UHII - Weekly Averages', folder='No-Cor/Ts/HII')
plot_hii(suhiiw, title='SUHII - Weekly Averages', folder='No-Cor/Ts/HII')

uhiic, suhiic = uhi_suhi(largec)
suhiicw = suhiic.resample('W').mean()
plot_hii(suhiicw, title='SUHII - Weekly Averages', folder='Cor-Factor/Ts/HII')

print(time.time() - start)
