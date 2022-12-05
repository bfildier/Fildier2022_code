#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Add lon/lat grid, date and HALO circle on GOES Images

Created on Fri Jun 25 10:28:32 2021

@author: bfildier
"""



import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import xarray as xr
import pandas as pd
import pytz
from datetime import datetime as dt
from datetime import timedelta, timezone
import sys,os,glob
import IPython
import argparse
import imageio
    
import pickle
from matplotlib import cm
# import matplotlib.image as mpimg
from math import ceil

# geodesic distances and displacements
import geopy
import geopy.distance
# map display
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Circle

# ## Graphical parameters
# plt.style.use(os.path.join(matplotlib.get_configdir(),'stylelib/presentation.mplstyle'))

##-- directories and modules

workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'snapshots/with_HALO_circle'
resultdir = os.path.join(repodir,'results',subdirname)
figdir = os.path.join(repodir,'figures',subdirname)
inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
inputdir_png = os.path.join(repodir,'images/patterns/PNG')

# Load own module
projectname = 'EUREC4A_organization'
# while os.path.basename(repodir) != projectname:
#     repodir = os.path.dirname(repodir)
thismodule = sys.modules[__name__]

## Own modules

sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *

##--- local functions

def defineSimDirectories():
        
    # create output directory if not there
    os.makedirs(os.path.join(resultdir),exist_ok=True)
    

if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Add grid, date, time and HALO circle on snapshot")
    parser.add_argument('--day', default='20200126',help="day of snapshot")
    parser.add_argument('--time', default='1400',help="time of snapshot")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    day = args.day
    time = args.time
    draw_circle=True
    # day = '20200126'
    # time = '1400'
    
    # define output directory
    defineSimDirectories()
    
    # load png image
    lat_box = 11,16
    lon_box = -60,-52
    # HALO circle coordinates
    lon_center = -57.717
    lat_center = 13.3
    lon_pt_circle = -57.245
    lat_pt_circle = 14.1903

    res = '50m' # for cartopy
    
    im_path = os.path.join(inputdir_png,'GOES16__%s_%s.png'%(day,time))
    im = imageio.imread(im_path)
    
    ##-- show image  and add things on it
    
    fig = plt.figure(figsize=(np.diff(lon_box),np.diff(lat_box)))
    gs = gridspec.GridSpec(1, 1, width_ratios=[1])
    
    land = cfeature.NaturalEarthFeature('physical', 'land', \
    scale=res, edgecolor='k', facecolor=cfeature.COLORS['land'])
    
    ax = plt.subplot(gs[0],projection=ccrs.PlateCarree())
    ax.coastlines(resolution=res)
    ax.imshow(im,extent=[*lon_box,*lat_box],origin='upper')
    ax.set_extent([*lon_box,*lat_box])
    # Barbados island
    ax.add_feature(land, facecolor='beige')
    # gridlines
    gl = ax.gridlines(color='Grey',draw_labels=False)
    # HALO circle
    if draw_circle:
        r_circle = np.sqrt((lon_pt_circle - lon_center) ** 2 +
                           (lat_pt_circle - lat_center) ** 2)
        circ = Circle((lon_center,
                       lat_center),
                      r_circle, linewidth=2,
                      ec='w',
                      fill=False)
        ax.add_patch(circ)
    # day and time
    ax.text(0, 1,
            pytz.utc.localize(dt.strptime(day+time,'%Y%m%d%H%M')).strftime("%Y-%m-%d %H:%M"), ha='left', va='top',
            color='white',fontsize=20, transform=ax.transAxes)
    
    plt.savefig(os.path.join(figdir,'GOES16__%s_%s.png'%(day,time)),bbox_inches='tight')
    
