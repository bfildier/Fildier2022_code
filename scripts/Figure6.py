#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3

Created on Thu Aug  4 08:19:28 2022

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


#%% Functions for connecting zoomed subplots

from matplotlib.transforms import Bbox, TransformedBbox, blended_transform_factory

from mpl_toolkits.axes_grid1.inset_locator import BboxPatch, BboxConnector,\
    BboxConnectorPatch

def connect_bbox(bbox1, bbox2,
                 loc1a, loc2a, loc1b, loc2b,
                 prop_connect, prop_boxes=None):
    
    if prop_boxes is None:
        prop_boxes = prop_connect.copy()
        prop_boxes["alpha"] = prop_boxes.get("alpha", 1)*0.2

    c1 = BboxConnector(bbox1, bbox2, loc1=loc1a, loc2=loc2a, **prop_connect)
    c1.set_clip_on(False)
    c2 = BboxConnector(bbox1, bbox2, loc1=loc1b, loc2=loc2b, **prop_connect)
    c2.set_clip_on(False)

    bbox_patch1 = BboxPatch(bbox1, **prop_boxes)
    bbox_patch2 = BboxPatch(bbox2, **prop_boxes)

    p = BboxConnectorPatch(bbox1, bbox2,
                           loc1a=loc1a, loc2a=loc2a, loc1b=loc1b, loc2b=loc2b,
                           **prop_connect)
    p.set_clip_on(False)

    return c1, c2, bbox_patch1, bbox_patch2, p

def zoom_effect_yaxis(ax1, ax2, **kwargs):
    """
    ax2 : the big main axes
    ax1 : the zoomed axes
    The xmin & xmax will be taken from the
    ax1.viewLim.
    """

    tt = ax1.transScale + (ax1.transLimits + ax2.transAxes)
    trans = blended_transform_factory(tt,ax2.transData)

    mybbox1 = ax1.bbox
    mybbox2 = TransformedBbox(ax1.viewLim, trans)

    prop_boxes = kwargs.copy()
    prop_boxes["ec"] = "darkgoldenrod"
    prop_boxes["fc"] = "cornsilk"
    prop_boxes["alpha"] = 0.2
    
    prop_connect = kwargs.copy()
    prop_connect["ec"] = "darkgoldenrod"
    prop_connect["linestyle"] = '-'
    prop_connect["linewidth"] = 0.8
    prop_connect["alpha"] = 0.2

    c1, c2, bbox_patch1, bbox_patch2, p = \
        connect_bbox(mybbox1, mybbox2,
                     loc1a=2, loc2a=1, loc1b=3, loc2b=4, 
                     prop_connect=prop_connect, prop_boxes=prop_boxes)

    ax1.add_patch(bbox_patch1)
    ax2.add_patch(bbox_patch2)
    ax2.add_patch(c1)
    ax2.add_patch(c2)
    ax2.add_patch(p)

    return c1, c2, bbox_patch1, bbox_patch2, p


def computeWPaboveZ(qv,pres,p_top):
    """Calculates the integrated water path above each level.

    Arguments:
        - qv: specific humidity in kg/kg, Nz-vector
        - pres: pressure coordinate in hPa, Nz vector
        - p_top: pressure of upper integration level

    returns:
        - wp_z: water path above each level, Nz-vector"""

    Np = qv.shape[0]
    wp_z = np.full(Np,np.nan)

    p_increasing = np.diff(pres)[0] > 0
    
    if p_increasing:
        
        i_p_top = np.where(pres >= p_top)[0][0]
        
        for i_p in range(i_p_top,Np):
        # self.wp_z[:,i_z] = self.mo.pressureIntegral(arr=data.specific_humidity[:,i_z:],pres=pres[i_z:],p_levmin=pres[i_z],p_levmax=pres[-1],z_axis=z_axis)

            arr = qv
            p = pres
            p0 = p_top
            p1 = p[i_p]
            i_w = i_p
            
            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    else:
        
        i_p_top = np.where(pres >= p_top)[0][-1]

        for i_p in range(i_p_top):
            
            arr = np.flip(qv)
            p = np.flip(pres)
            p0 = p_top
            p1 = pres[i_p]
            i_w = i_p

            wp_z[i_w] = mo.pressureIntegral(arr=arr,pres=p,p0=p0,p1=p1)

    return wp_z



#%% Figure 6

i_fig = 6

hPa_to_Pa = 1e2

#--- data

Nsample_W = 20
Nsample_H = 15
inds_uniformRH = slice(6,26)

# coordinates
W_all = [float(str(radprf_MI_20200213lower.name[inds_uniformRH][i].data)[2:6]) for i in range(Nsample_W)]
# W_all = np.array(W_all)-W_ref
H_all = [float(str(radprf_MI_20200213lower.name[inds_uniformRH.stop:inds_uniformRH.stop+Nsample_W][i].data)[11:15]) for i in range(Nsample_H)]


i_ref = 1
radprf2show = radprf_MI_20200213lower

# peak height
# z_jump = 1.66883978
z_jump = 1.81
z = radprf2show.zlay[0]/1e3 # km
k_jump = i_jump = np.where(z>=z_jump)[0][0]
pres_jump = radprf2show.play[k_jump]/1e2 # hPa

qradlw_peak = np.full((Nsample_W,Nsample_H),np.nan)
qradlw_peak_ref = np.full((Nsample_W,Nsample_H),np.nan)

delta_qradlw_peak = np.full((Nsample_W,Nsample_H),np.nan)

ratio_qradlw_peak = np.full((Nsample_W,Nsample_H),np.nan)

# mask intrusions spanning the 10km level
is_valid = np.full((Nsample_W,Nsample_H),True)

for i_W in range(Nsample_W):
    for i_H in range(Nsample_H):
        
        W = W_all[i_W]
        H = H_all[i_H]
        name = 'W_%2.2fmm_H_%1.2fkm'%(W,H)
        i_prof = np.where(np.isin(radprf2show.name.data,name))[0][0]
        
        qradlw_peak[i_W,i_H] = radprf_MI_20200213lower.q_rad_lw[i_prof,k_jump].data
        qradlw_peak_ref[i_W,i_H] = radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        
        delta_qradlw_peak[i_W,i_H] = radprf_MI_20200213lower.q_rad_lw[i_prof,k_jump].data - radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        
        ratio_qradlw_peak[i_W,i_H] = radprf_MI_20200213lower.q_rad_lw[i_prof,k_jump].data / radprf_MI_20200213lower['q_rad_lw'].data[i_ref].data[i_jump]
        
        # Mark False if idealized intrusion touches the upper bound at 10km 
        i_int_max = np.where(radprf_MI_20200213lower['rh'].data[i_prof] == radprf_MI_20200213lower['rh'].data[i_prof][0])[0][-1]
        i_top = np.where(z <= 10)[0][-1]
        if i_int_max < i_top: # if moist intrusion is all below 10km, mark True
            is_valid[i_W,i_H] = False
            
#--- show

# Figure layout
fig = plt.figure(figsize=(13.5,4.5))

gs = GridSpec(1, 3, width_ratios=[1.5,1.5,2.5], height_ratios=[1],hspace=0.25,wspace=0.3)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# peak height
# z_jump = 1.66883978
z_jump = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][0]
z = radprf_MI_20200213.zlay[0]/1e3 # km
k_jump = np.where(z>=z_jump)[0][0]
pres_jump = radprf_MI_20200213.play[k_jump].data/1e2 # hPa

labs = 'no intrusion','fitted from data','homogenized, same W','idealized, same W'

#-- (a) and (b)
# radprf_ab = radprf_MI_20200213
# i_ref = 4
# i_mi = 2
# i_rectRH = 4
# i_homoRH = 3
radprf = radprf_MI_20200213lower
i_ref = 1
i_mi = 2
i_rectRH = 3
i_homoRH = 4

for ax, varid in zip((ax1,ax2),('rh','q_rad_lw')):
    
    #- peak level
    ax.axhline(z_jump,c='darkgoldenrod',linestyle='--',linewidth=0.8,alpha=0.8)
        
    #- reference with intrusion (from piecewise linear fit)
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_mi = radprf[varid].data[i_mi]
    # show
    if ax == ax1:
        lab = labs[1]
    ax.plot(var_mi,z,'b',label=lab)

    #- reference without intrusion (from piecewise linear fit)
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_ref = radprf[varid].data[i_ref]
    # show
    if ax == ax1:
        lab = labs[0]
    ax.plot(var_ref,z,'grey',label=lab)
    
    #- rectangle-RH intrusion, same water path, same height
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_rectRH = radprf[varid].data[i_rectRH]
    # show
    if ax == ax1:
        lab = labs[3]
    ax.plot(var_rectRH,z,'k',label=lab)
    
    #- homogenized rh, same water path at peak level
    z = np.array(radprf.zlay[i_ref]/1e3) # km
    var_homoRH = radprf[varid].data[i_homoRH]
    # show
    if ax == ax1:
        lab = labs[2]
    ax.plot(var_homoRH,z,'k--',label=lab)
    


ax1.set_ylabel('z (km)')
ax2.set_ylabel('z (km)')
ax1.set_xlabel('Relative humidity')
ax2.set_xlabel(r'LW $Q_{rad}$ (K/day)')

ax1.set_ylim((-0.15,10.15))
ax2.set_ylim((0,2.3))
ax2.set_xlim((-13.1,0.1))

ax1.legend(fontsize=7,loc='upper right')

#- connecting (a) and (b)
zoom_effect_yaxis(ax2, ax1)


#-- (c) intrusions at all heights and water paths
ax = ax3
# radprf = radprf_RMI_20200213
radprf = radprf_MI_20200213lower

# cmap = plt.cm.RdYlBu_r
cmap = plt.cm.RdYlBu
vmin = 0.
vmax = 1.
norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

# data
z = np.ma.masked_array(ratio_qradlw_peak,is_valid).T
# z = np.ma.masked_array(qradlw_peak,is_valid).T

# colors
ax.contourf(W_all,H_all,z,levels=30,cmap=cmap,vmin=vmin,vmax=vmax)

# lines
cont = ax.contour(W_all,H_all,z,levels=np.linspace(0.1,0.9,9),colors=('grey',),
                  linestyles=('-',),linewidths=(0.8,),vmin=vmin,vmax=vmax)
plt.clabel(cont, fmt = '%1.1f', colors = 'grey', fontsize=10) #contour line labels

# labels
ax.set_xlabel('Intrusion water path (mm)')
ax.set_ylabel('Intrusion height (km)')

# colorbar
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),
                 ax=ax,shrink=0.99,pad=0.04)
# cb.set_label(r'LW $Q_{rad}$ peak (K/day)')
cb.set_label(r'Ratio to reference peak')

ax.set_title(r'Normalized lower peak (at 1.8 km)')

colors = 'orange','blue','green','purple','red','brown'

# Each day's intrusion
for day_col,daylab in zip(colors,moist_intrusions.keys()):
    
    W_int = moist_intrusions[daylab]['stats']['W_int']
    z_int_bottom = moist_intrusions[daylab]['stats']['z_int_bottom']
    z_int_center = moist_intrusions[daylab]['stats']['z_int_center']
    z_int_top = moist_intrusions[daylab]['stats']['z_int_top']
    
    # key_col = 'cornsilk'
    # key_col = 'k'
    
    if daylab in ['20200211']:
        ax.scatter(W_int,z_int_center,marker='o',c=day_col,edgecolor='none')
        ax.text(W_int+0.32,z_int_center,' '+daylab,c=day_col,ha='right',va='bottom',rotation='vertical')
    elif daylab in ['20200209','20200128','20200213, upper']:
        ax.scatter(W_int,z_int_center,marker='o',c=day_col,edgecolor='none')
        ax.text(W_int,z_int_center,' '+daylab,c=day_col,ha='right',va='bottom',rotation='vertical')
    else:
        ax.scatter(W_int,z_int_center,marker='o',c='blue',edgecolor='blue')
        ax.text(W_int,z_int_center,' '+daylab,c=day_col,ha='right',va='bottom',rotation='vertical')
    
# # Equivalent height of cooling peak reduction by a uniform increas in RH
# # (the center height of a rectangle intrusion that gives the same peak reduction)
# H_equiv = np.zeros((Nsample,))
# for i_W in range(Nsample):
        
#     W = W_all[i_W]#+W_ref
#     name = 'W_%2.2fmm_uniform_RH'%(W)
#     i_prof = np.where(np.isin(radprf.name.data,name))[0][0]
#     qradlw_peak_uRH = radprf.q_rad_lw[i_prof,k_jump].data
#     # print(i_prof,qradlw_peak_uRH)
#     delta_qradlw_uRH = qradlw_peak_uRH - qradlw_peak_ref
    
#     i_H = np.where(delta_qradlw_peak[i_W,:] >= delta_qradlw_uRH)[0][0]
#     H_equiv[i_W] = H_all[i_H]

# ax.plot(W_all,H_equiv,'r')

# # Center of mass of a uniform free-tropospheric RH (only depends on the shape of qvstar)
# i_levmax = np.where(~np.isnan(qvstar))[0][-1] # assuming data is ordered from bottom to top
# p_levmax = pres[i_levmax]
# W_qvstar = computeWPaboveZ(qvstar,pres,p_levmax)
# z_jump_FT = moist_intrusions['20200213, lower']['fit']['z_breaks_id'][1]
# i_jump_FT = np.where(z>z_jump_FT)[0][0]
# i_z = np.where(W_qvstar >= W_qvstar[i_jump_FT]/2)[0][0]
# z_center_uRH = z[i_z]
# i_H = np.where(H_all >= z_center_uRH)[0][0]
# H_qvstar_center = H_all[i_H]
# H_uniform = H_qvstar_center*np.ones((Nsample,))

# ax.plot(W_all,H_uniform,'r')


# labels
ax.set_xlabel('Intrusion water path (mm)')
ax.set_ylabel('Intrusion center level (km)')
ax.set_ylim([H_all[0],H_all[-1]])


#--- Add panel labeling
pan_labs = '(a)','(b)','(c)'
pan_cols = 'k','k','k'
axs = ax1,ax2,ax3
for ax,pan_lab,pan_col in zip(axs,pan_labs,pan_cols):
    t = ax.text(0.04,0.02,pan_lab,c=pan_col,ha='left',va='bottom',
            transform=ax.transAxes,fontsize=14)
    # if ax != ax3:
        # t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))

#--- save
plt.savefig(os.path.join(figdir,'Figure%d.pdf'%i_fig),bbox_inches='tight')
# plt.savefig(os.path.join(figdir,'Figure%d.png'%i_fig),dpi=300,bbox_inches='tight')