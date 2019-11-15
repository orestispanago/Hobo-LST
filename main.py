import os
import pandas as pd
from datareader import Hobo
from datareader import dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import statsmodels.formula.api as sm
from statsmodels.api import add_constant
from pysolar import solar

# Gets rid of pandas FutureWarning
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

params = {'figure.figsize': (14, 4),
          'axes.titlesize': 18,
          'axes.titleweight': 'bold',
          'axes.labelsize': 18,
          'axes.labelweight': 'bold',
          'xtick.labelsize': 18,
          'ytick.labelsize': 18,
          'font.weight' : 'bold',
          'font.size': 18,
          'savefig.dpi': 300.0,
          'savefig.format': 'tiff',
          'figure.constrained_layout.use': True}
plt.rcParams.update(params)


def plot_hobo_modis(df, folder=None):
    for station in Hobo.stations:
#        plt.figure(figsize=(14, 4))
        plt.plot(df[station.alias + 'T'])
        plt.plot(df[station.alias])
        plt.title(station.alias)
        plt.legend(('Hobo', 'MODIS'))
        plt.ylabel('Temperature $(\degree$C)')
        plt.xlabel('Datetime - UTC')
        if folder is not None:
            plt.savefig(make_dir(folder) + station.alias)
        plt.show()

def plot_hobo_modis_resids(df, folder=None):
    for station in Hobo.stations:
#        plt.figure(figsize=(14, 4))
        plt.plot(df[f"{station.alias}T-{station.alias}"])
        plt.title(station.alias)
        plt.legend(['Hobo-MODIS'])
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

        x = add_constant(ref)  # include constant (intercept) in ols model
        mod = sm.OLS(test, x)
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
#    plt.figure(figsize=(14, 4))
    for i in others:
        plt.plot(df[i.alias])
        plt.legend(bbox_to_anchor=(1, 1))
    plt.title(title)
    plt.tight_layout()
    if folder is not None:
        plt.savefig(make_dir(folder) + title)
    plt.show()


def cor_modis(df):
    """ Applies correction factor to MODIS data based on ground stations"""
    for i in regresults.index:
        slope = regresults.loc[i]['slope']
        intercept = regresults.loc[i]['intercept']
        df[i] = dataset[i] / slope - intercept
    return df


def plot_heatmap(df, title=None, folder=None):
    byday = dow(df)
    fig, ax = plt.subplots(figsize=(10,10))
    heatmap = sns.heatmap(byday,annot=True,ax=ax,cbar_kws={'label': 'UHII ($\degree$C)'})
    
    if folder is not None:
        plt.title(title)
        fig = heatmap.get_figure()
        fig.savefig(make_dir(folder) + title)
    plt.show()


def plot_diurnal(df, title=None, folder=None):
    hourly = df.groupby(df.index.hour).mean()
    for o in others:
        plt.plot(hourly[o.alias])
        plt.legend(bbox_to_anchor=(1, 1.07))
        plt.xlabel('Time of Day')
    if folder is not None:
        plt.title(title)
        plt.tight_layout()
        plt.savefig(make_dir(folder) + title)
    plt.show()


def dow(df):
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']
    by_dow = df.groupby(df.index.weekday_name).mean().reindex(weekdays)
    by_dow.index=[i[:3] for i in weekdays]
    return by_dow


def plot_dow(df, title=None, folder=None):
    byday = dow(df)
    fig = plt.figure(figsize=(12,6))
    ax = plt.subplot(111)
    for o in others:
        ax.plot(byday[o.alias])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Day of week')
#    ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    if folder is not None:
        ax.set_title(title)
        fig.savefig(make_dir(folder) + title)
    plt.show()


def zenith(df):
    """ Calculates solar zenith for dataframe index and adds new zen column"""
    dfzen = df.copy()
    zenlist = []
    for t in df.index:
        zen = solar.get_altitude(38.291969, 21.788156, t)
        zenlist.append(zen)
    dfzen['zen'] = zenlist
    return dfzen

def day_night(df,day=True):
    """ Returns day/night dataframe by masking night/day rows with NaN"""
    dfout= zenith(df)
    if day is True:
        dfout = dfout.mask(dfout['zen'] < 1.0)
    else:
        dfout =  dfout.mask((dfout['zen'] < 89.0) & (dfout['zen'] > 1.0))
    return dfout

def calc_resids(df):
    for station in Hobo.stations:
        df[f"{station.alias}T-{station.alias}"]= df[station.alias + 'T'] - df[station.alias]
    return df

ref_hobo = Hobo.stations[0]
others = [station for station in Hobo.stations[1:]]

regresults = reg_stats(dataset)
regresults.to_excel(make_dir('No-Cor/Reg') + 'results.xlsx')
# plot_scatters(dataset, folder='No-Cor/Reg', name='hobo_modis_scatter.png')

corrected = cor_modis(dataset.copy())
regresultsc = reg_stats(corrected)
regresultsc.to_excel(make_dir('Cor-Factor/Reg') + 'results.xlsx')
# plot_scatters(corrected, folder='Cor-Factor/Reg', name='hobo_modis_scatter.png')

daily = dataset.resample('D').mean()
daily = calc_resids(daily)
#plot_hobo_modis_resids(daily,folder='No-Cor/Ts/Daily/Resids')
# plot_hobo_modis(daily, folder='No-Cor/Ts/Daily')

dailyc = corrected.resample('D').mean()
dailyc = calc_resids(dailyc)
#plot_hobo_modis_resids(dailyc,folder='Cor-Factor/Ts/Daily/Resids')
# plot_hobo_modis(dailyc, folder='Cor-Factor/Ts/Daily')

uhii, suhii = uhi_suhi(dataset)
plot_hii(uhii.resample('W').mean()[5:], title='UHII - Weekly Averages', folder='No-Cor/Ts/HII')
# plot_hii(suhii.resample('W').mean()[5:], title='SUHII - Weekly Averages', folder='No-Cor/Ts/HII')

uhiic, suhiic = uhi_suhi(corrected)
# plot_hii(suhiic.resample('W').mean()[5:], title='SUHII - Weekly Averages', folder='Cor-Factor/Ts/HII')

plot_heatmap(uhii, title='UHII', folder='No-Cor/Ts/HII')
plot_heatmap(suhii, title='SUHII - No correction', folder='No-Cor/Ts/HII')
plot_heatmap(suhiic, title='SUHII- Corrected', folder='Cor-Factor/Ts/HII')


plot_diurnal(uhii, title='UHII - Diurnal variation', folder='No-Cor/Ts/HII')

plot_dow(uhii, title='UHII - DoW', folder='No-Cor/Ts/HII')


uhii_day = day_night(uhii,day=True)
uhii_night = day_night(uhii,day=False)
hourly_day = uhii_day.groupby(uhii_day.index.hour).mean()

#suhii_day = day_night(suhii,day=True)
#plot_hii(suhii_day.resample('W').mean()[5:])
#suhii_night = day_night(suhii,day=False)
#plot_hii(suhii_night.resample('W').mean()[5:])
