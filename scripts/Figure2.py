#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2

Created on Fri Jul 29 13:38:32 2022

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
    
    #%% Figure S2 -- spectral simplifications
    
    i_fig = 2

    fig,axs = plt.subplots(ncols=1,nrows=2,figsize=(6,8))

    ##-- (a) spectral fit
    ax = axs[0]

    wn_range = kappa_data['nu']
    slice_fit = kappa_data['s_fit_rot']
    nu_fit = wn_range[slice_fit]
    kappa_nu = kappa_data['kappa']
    kappa_fit = kappa_data['kappa_rot_fit']

    nu_star = 554 # cm-1
    delta_nu = 160.1 # cm-1

    ax.fill_betweenx(x1=[nu_star-delta_nu/2]*2,x2=[nu_star+delta_nu/2]*2,y=[1e-5,3e5],color='k',alpha=0.05)
    ax.axvline(x=nu_star,c='k',linewidth=0.5,linestyle=':')

    ax.plot(wn_range,kappa_nu,linewidth=0.1,c='mediumblue')
    ax.plot(wn_range[slice_fit],kappa_fit,linewidth=1,c='k',linestyle='--',)

    # dot
    ax.scatter(x=nu_star,y=1/3,s=20,color='k')

    ax.set_yscale('log')
    ax.set_ylabel(r'Extinction coefficient $\kappa$ (m2/kg)')
    ax.set_xlabel(r'Wavenumber $\tilde{\nu}$ (cm$^{-1}$)')

    ax.set_ylim((1e-5,3e5))

    ##-- (b) spectral widths (phi(nu) & Planck(nu))
    ax = axs[1]

    rs = rad_scaling_all['20200202']

    Ws = 0.5,3,24
    # colors
    cols = ['mediumvioletred','purple','k']

    for i_W in range(len(Ws)):
        
        W = Ws[i_W]
        phi = rs.phi(kappa_fit,W)
        # ax.plot(wn_range[slice_fit],phi,c=cols[i_W][:3]/255,label=r"$\phi_\nu$(W=%2.1fmm)"%W)
        ax.plot(wn_range[slice_fit],phi,c=cols[i_W],label=r"$\phi_\nu$(W=%2.1fmm)"%W)

    # center of reference curve        
    ax.axvline(x=nu_star,c='k',linewidth=0.5,linestyle=':')
        
    ax.set_xlabel(r'Wavenumber $\tilde{\nu}$ (cm$^{-1}$)')
    ax.set_ylabel(r'Weighting function $\phi_\nu$')
    ax.legend(loc='upper right')


    ax_B = ax.twinx()

    B_nu_300 = rs.planck(wn_range*100,300)*1e2 # W.sr-1.m-2.cm 
    B_nu_290 = rs.planck(wn_range*100,290)*1e2 # W.sr-1.m-2.cm 
    B_nu_280 = rs.planck(wn_range*100,280)*1e2 # W.sr-1.m-2.cm 

    ax_B.plot(wn_range,pi*B_nu_290,'r',label=r'$\pi B_\nu$(T=290K)')
    ax_B.fill_between(wn_range,y1=pi*B_nu_280,y2=pi*B_nu_300,color='r',edgecolor='none',alpha=0.1,label=r'$\pi B_\nu(\pm$ 1K)')

    # ax_B.plot(wn_range,pi*B_nu_280,'k:',label=r'$\pi B_\nu$(T=280K)')

    ax_B.legend(loc='right')
    ax_B.set_ylabel(r'Planck $\pi B_\nu$ (J.s$^{-1}$.sr$^{-1}$.m$^{-2}$.cm)')

    # align zero of both graphs
    ax_lim = ax.axes.get_ylim()
    ax_frac_below_zero = -ax_lim[0]/(ax_lim[1]-ax_lim[0])
    ax_B_range = 0.65
    ax_B.set_ylim((-ax_frac_below_zero*ax_B_range,(1-ax_frac_below_zero)*ax_B_range))

    #--- Add panel labeling
    pan_labs = ["(%s)"%chr(i) for i in range(ord('a'), ord('b')+1)]
    for ax,pan_lab in zip(axs.flatten(),pan_labs):
        t = ax.text(0.04,0.91,pan_lab,transform=ax.transAxes,fontsize=14)
        t.set_bbox(dict(facecolor='w',alpha=0.8,edgecolor='none'))

    #--- save
    # plt.savefig(os.path.join(figdir,'FigureS%d.pdf'%i_fig),bbox_inches='tight')
    plt.savefig(os.path.join(figdir,'FigureS%d.png'%i_fig),dpi=300,bbox_inches='tight')
