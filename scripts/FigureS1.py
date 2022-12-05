#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S1

Created on Fri Jul 29 13:28:03 2022

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
import argparse

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


##--- local functions

from radiativefeatures import *
from radiativescaling import *
# from thermodynamics import *
from conditionalstats import *
from matrixoperators import *
from thermoConstants import *
from thermoFunctions import *

mo = MatrixOperators()

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
    
    #%% Figure S1 -- Diurnal cycle
    
    i_fig = 1
    
    rad_range = 'lw'
    rad_labels = {'net':'',
                  'sw':'SW',
                  'lw':'LW'}
    rad_titles = {'net':'Net heating',
                  'sw':'Shortwave',
                  'lw':'Longwave'}
    
    PW_lims = [20,50] # km
    
    # colors
    cmap = plt.cm.ocean_r
    vmin = PW_lims[0]
    vmax = PW_lims[1]
    
    def defineCol(var_col,cmap,vmin,vmax):
        
        norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
        scmap = cm.ScalarMappable(norm=norm, cmap=cmap)
        cols = cmap(norm(var_col),bytes=True)/255 
    
        return cols,scmap
    
    ##-- plot
    
    fig,axs = plt.subplots(ncols=3,nrows=2,figsize=(13,8))
    
    x_left = np.nan
    x_right = np.nan
    y_bot = np.nan
    y_top = np.nan
    
    days2show = ['20200122','20200202']
    
    for day,axs_row in zip(days2show,axs):
    
        print('--',day)
        
        f = rad_features_all[day]
        var_col = f.pw
        cols,scmap = defineCol(var_col,cmap,vmin,vmax)
    
        date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
        day_str = date.strftime("%Y-%m-%d")
        t = np.array([(pytz.utc.localize(dt.strptime(str(t)[:19],"%Y-%m-%dT%H:%M:%S")) - date).seconds/3600 for t in f.launch_time])
        
        for ax,rad_range in zip(axs_row,list(rad_labels.keys())):
            
            print('-',rad_range)
            
            for i_lt in range(f.launch_time.size):
            
                # print(i_lt,end='..')
                x = t[i_lt]
                y = getattr(f,'qrad_%s_smooth'%rad_range)[i_lt,f.i_lw_peak[i_lt]]
                c = f.pw[i_lt]
            
                ax.scatter(x=x,y=y,c=[cols[i_lt][:3]],vmin=vmin,vmax=vmax)
            
            # titles
            if day == days2show[0]:
                ax.set_title(rad_titles[rad_range])
            # x labels
            if day == days2show[1]:
                ax.set_xlabel('Time of day (hr)')
            # y labels
            if rad_range == 'net':
                ax.set_ylabel('Cooling (K/day) on day %s'%day_str)
    
                
            # Save boundaries for legend
            x,y,w,h = ax.get_position().bounds
            x_left = np.nanmin(np.array([x,x_left]))
            x_right = np.nanmax(np.array([x+w,x_right]))
            y_bot = np.nanmin(np.array([y,y_bot]))
            y_top = np.nanmax(np.array([y+h,y_top]))
    
    # Color bar
    dx = (x_right-x_left)/60
    cax = plt.axes([x_right+2*dx,y_bot,dx,y_top-y_bot])
    cbar = fig.colorbar(scmap, cax=cax,orientation='vertical')
    cbar.ax.set_ylabel('PW (mm)',fontsize=14)
    
    #--- Add panel labeling
    pan_labs = ["(%s)"%chr(i) for i in range(ord('a'), ord('f')+1)]
    for ax,pan_lab in zip(axs.flatten(),pan_labs):
        t = ax.text(0.04,0.91,pan_lab,transform=ax.transAxes,fontsize=14)
        t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))
    
    #--- save
    plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
    # plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')
    
    
    
    
