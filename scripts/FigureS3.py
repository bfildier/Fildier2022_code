#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S3

Created on Sun Jul 17 16:43:31 2022

@author: bfildier
"""

##-- modules

# general
import scipy.io
import sys, os, glob
import numpy as np
import xarray as xr
from datetime import datetime as dt
from datetime import timedelta, timezone
import pytz
import pickle

# stats
from scipy.stats import gaussian_kde
from scipy.stats import linregress
from scipy import optimize

# images
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Circle
from PIL import Image
import matplotlib.image as mpimg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

##-- directories

# workdir = os.path.dirname(os.path.realpath(__file__))
workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
resultdir = os.path.join(repodir,'results','radiative_features')
figdir = os.path.join(repodir,'figures','paper')
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
radinputdir = os.path.join(repodir,'input')
imagedir = os.path.join(repodir,'figures','snapshots','with_HALO_circle')
scriptsubdir = 'Fildier2021'

# Load own module
projectname = 'EUREC4A_organization'
thismodule = sys.modules[__name__]

## Own modules
sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from radiativescaling import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *
from thermoConstants import *
from thermoFunctions import *

mo = MatrixOperators()

##--- local functions

def defineSimDirectories():
    """Create specific subdirectories"""
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir),exist_ok=True)
    
    
if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Draw paper figures from all precomputed data")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)

    # output directory
    defineSimDirectories()
    
    ##-- Load all data
    
    exec(open(os.path.join(workdir,"load_data.py")).read())
    
#%% --- Figure S3 Fish vs, Flower example

i_fig = 3

fig,axs = plt.subplots(ncols=2,figsize=(8,5))

pw_max = 45 # mm
z_max = 3.2 # km
rh_cloud = 0.95

qrad_clear_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}
iscloud_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}
rh_clear_by_pattern = {'Fish':[],
                   'Flower':[],
                   'Gravel':[],
                   'Sugar':[]}

for day,pat,conf in zip(days,name_pattern,confidence_pattern):
    
    pw = rad_features_all[day].pw
    z_peak = rad_features_all[day].z_lw_peak/1e3 # km
    qrad_peak = rad_features_all[day].qrad_lw_peak
    keep_large = qrad_peak < -5 # K/day
    lon_day = data_all.sel(launch_time=day).longitude[:,50]
    lat_day = data_all.sel(launch_time=day).latitude[:,50]
    keep_box = np.logical_and(lon_day < lon_box[1], lat_day >= lat_box[0])
    keep_for_panel_e = np.logical_and(keep_large,keep_box)
    
    keep_low = z_peak < z_max # km
    keep_dry = pw < pw_max
    keep_subset = np.logical_and(keep_low,keep_dry)
    
    k = np.logical_and(keep_for_panel_e,keep_subset)
    
    # is cloud ?
    data_day = data_all.sel(launch_time=day)
    iscloud_day = np.any(data_day.relative_humidity > rh_cloud,axis=1).data
    iscloud_by_pattern[pat].extend(list(iscloud_day[k]))
    
    # rh
    rh_prof = data_day.relative_humidity.data
    rh_save = np.array([rh_prof[:,i_z][k] for i_z in range(rh_prof.shape[1])])
    rh_save  = np.swapaxes(rh_save,0,1)
    rh_clear_by_pattern[pat].append(rh_save)
    
    # qrad profile
    qrad_prof = data_day.q_rad_lw.data
    qrad_save = np.array([qrad_prof[:,i_z][k] for i_z in range(qrad_prof.shape[1])])
    qrad_save  = np.swapaxes(qrad_save,0,1)
    qrad_clear_by_pattern[pat].append(qrad_save)

for pat in ['Fish','Flower']:
    
    isclear = np.logical_not(iscloud_by_pattern[pat])
    
    # rh
    ax = axs[0]
    
    rh_prof = np.vstack(rh_clear_by_pattern[pat])
    rh_clear = np.array([rh_prof[:,i_z][isclear] for i_z in range(rh_prof.shape[1])])
    rh_clear = np.swapaxes(rh_clear,0,1)
    
    rh_5,rh_50,rh_95 = np.nanpercentile(rh_clear,[5,50,95],axis=0)
    
    ax.fill_betweenx(y=z,x1=rh_5,x2=rh_95,color=col_pattern[pat],edgecolor=None,
                      alpha=0.12)
    ax.plot(rh_50,z,c=col_pattern[pat],alpha=0.5)
    for i_p in range(rh_clear.shape[0]):
        ax.plot(rh_clear[i_p],z,c=col_pattern[pat],alpha=0.08,linewidth=0.6)
    
    # qrad
    ax = axs[1]
    
    qrad_prof = np.vstack(qrad_clear_by_pattern[pat])
    qrad_clear = np.array([qrad_prof[:,i_z][isclear] for i_z in range(qrad_prof.shape[1])])
    qrad_clear = np.swapaxes(qrad_clear,0,1)
    
    qrad_5,qrad_50,qrad_95 = np.nanpercentile(qrad_clear,[5,50,95],axis=0)
    
    ax.fill_betweenx(y=z,x1=qrad_5,x2=qrad_95,color=col_pattern[pat],edgecolor=None,
                      alpha=0.12)
    ax.plot(qrad_50,z,c=col_pattern[pat],alpha=0.5)
    for i_p in range(rh_clear.shape[0]):
        ax.plot(qrad_clear[i_p],z,c=col_pattern[pat],alpha=0.08,linewidth=0.6)
    

axs[1].set_xlim((-16,6))
axs[0].set_ylabel('z (km)') 
axs[0].set_xlabel('Relative humidity')
axs[1].set_xlabel('Longwave cooling (K/day)')

#--- Add panel labeling
pan_labs = '(a)','(b)'
pan_cols = 'k','k'
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    t = ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
            transform=ax.transAxes,fontsize=16)
    t.set_bbox(dict(facecolor='w', alpha=0.5, edgecolor='none'))

#--- save
# plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')
