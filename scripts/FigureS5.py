#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S5

Created on Wed Aug  3 11:00:49 2022

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
    

#%% Figure S5 -- behavior to warming

i_fig = 5

from itertools import product

SSTs = np.arange(296,342,2)

Nprof = len(SSTs)


radprf = radprf_warming
z = radprf.zlay[0] # km

#-- Show

varids = 'tlay','h2o','q_rad_lw'
Nv = len(varids)+1
linestyles = '-','-'
linewidths = 2,1

# cols
var_col = np.array(SSTs,dtype=float)
norm = matplotlib.colors.Normalize(vmin=SSTs[0], vmax=SSTs[-1])
cmap = plt.cm.inferno_r
# cols = cmap(norm(var_col),bytes=True)/255
cols = cmap(norm(var_col),bytes=True)/255

fig,axs_grd = plt.subplots(nrows=2,ncols=2,figsize=(9,9))
axs = axs_grd.flatten()

# fig = plt.figure(figsize=(4*Nv,5.5))

# gs = GridSpec(1, 3, width_ratios=[3,3,1], height_ratios=[1],hspace=0.25,wspace=0.3)
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1])
# ax3 = fig.add_subplot(gs[2])

# axs = np.array([ax1,ax2])

#-- RH
ax = axs[0]

ax.plot(radprf.rh[0],z,'b')
# value at 300K
SST = 300 # K
name = 'RHid_SST%d'%SST
i_p_file = np.where(radprf.name.data == name)[0][0]
var = radprf.rh.data[i_p_file,:]

ax.plot(radprf.rh[i_p_file],z,'k')


#-- T, qv and qrad LW
for i_ax in range(1,Nv):
    
    ax = axs.flatten()[i_ax]
    varid = varids[i_ax-1]
    
    ax.plot(radprf[varid].data[0],z,c='b',linewidth=linewidths[0],linestyle=linestyles[0])
    
    for i_prof in range(Nprof):
        
        SST = SSTs[i_prof]
        name = 'RHid_SST%d'%SST
        i_p_file = np.where(radprf.name.data == name)[0][0]
        
        var = radprf[varid].data[i_p_file,:]
        col = cols[i_prof]
        
        ax.plot(var,z,c=col,linewidth=linewidths[1],linestyle=linestyles[1],alpha=0.5)

# colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, ax=axs.ravel().tolist(),aspect=35)
cbar.set_label('SST (K)')

for ax in axs[:2]:
    ax.set_ylim((-0.1,9.1))
    
for ax in axs[2:]:
    ax.set_ylim((-0.1,4.1))


axs[1].set_xlim((238,342))
axs[2].set_xscale('log')
axs[2].set_xlim((3e-4,6e-1))
axs[3].set_xlim((-35.3,5.3))

axs[0].set_xlabel('Relative humidity')
axs[1].set_xlabel('Temperature (K)')        
axs[2].set_xlabel('Specific humidity (kg/kg)')
axs[3].set_xlabel('LW cooling (K/day)')

axs[0].set_ylabel('z (km)')
axs[2].set_ylabel('z (km)')

# suptitle at midpoint of left and right x-positions
# mid = (fig.subplotpars.right + fig.subplotpars.left)/2
# plt.suptitle(r'Varying intrusion height at $W=%1.2f$ mm'%W,fontsize=15,x=mid)

#--- Add panel labeling
pan_labs = '(a)','(b)','(c)','(d)'
pan_cols = 'k','k','k','k'
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    t = ax.text(0.02,0.98,pan_lab,c=pan_col,ha='left',va='top',
            transform=ax.transAxes,fontsize=16)
    t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='none'))

#--- save
plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
# plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')