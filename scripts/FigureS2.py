#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure S2

Created on Wed Aug  3 10:48:50 2022

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
    

#%% --- Figure S2 Pattern labeling

i_fig = 2

fig,axs = plt.subplots(ncols=3,nrows=4,figsize=(12,11),
                       subplot_kw={'projection':ccrs.PlateCarree()})
fig.subplots_adjust(wspace = 0.05, hspace = 0)


for i_sub in range(11):
    
    ax = axs.flatten()[i_sub]
    day_p = days[i_sub]
    name_p = name_pattern[i_sub]
    conf_p = confidence_pattern[i_sub]
    
    if day_p == '20200126':
        date = dt.strptime(day_p,'%Y%m%d')
        time = '1400'
        indir_goes_images = '/Users/bfildier/Data/satellite/GOES/images/%s'%date.strftime('%Y_%m_%d')
        image_vis_file = os.path.join(indir_goes_images,'C02_GOES_M2_10N-16N-60W-52W_%s_%s.jpg'%(day_p,time))
        image = Image.open(image_vis_file)
    else:
        image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
    
    # image = Image.open(os.path.join(workdir,'../images/patterns/PNG','GOES16__%s_1400.png'%day_p))
    
    
    # if len(glob.glob(os.path.join(workdir,'../images/patterns/PNG_brighter','GOES16__%s_1400.png'%day_p))) > 0:
    # image = Image.open(os.path.join(workdir,'../images/patterns/PNG_brighter','GOES16__%s_1400.png'%day_p))
        
    
    ax.imshow(np.asarray(image),extent=[*lon_box,*lat_box],origin='upper')
    # HALO circle
    circ = Circle((lon_center,
                   lat_center),
                  r_circle, linewidth=0.8,
                  ec='w',
                  fill=False)
    ax.add_patch(circ)
    # Barbados island
    res = '50m'
    land = cfeature.NaturalEarthFeature('physical', 'land', \
                                        scale=res, edgecolor='k', \
                                        facecolor=cfeature.COLORS['land'])
    ax.add_feature(land, facecolor='beige')
    
    #- add pattern name
    # ax.text(0.98,0.03,name_p,c='w',ha='right',transform=ax.transAxes,fontsize=14)
    #- add date
    text_str = day_p[:4]+'-'+day_p[4:6]+'-'+day_p[6:8]+'\n'+name_p+'\n'+conf_p
    ax.text(0.98,0.98,text_str,c='w',ha='right',va='top',transform=ax.transAxes,fontsize=12)
    # ax.outline_patch.set_visible(False)

    # change frame color
    rect = mpatches.Rectangle((0,0), width=1, height=1,edgecolor=col_pattern[name_p], facecolor="none",linewidth=3,alpha=1, transform=ax.transAxes)
    ax.add_patch(rect)
    

# remove last subplot
ax = axs.flatten()[-1]
fig.delaxes(ax)


#--- save
# plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=180,bbox_inches='tight')
