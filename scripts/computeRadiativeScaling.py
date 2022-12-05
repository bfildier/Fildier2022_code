#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computes all components of the radiative scaling for each day.

Created on Tue Aug 17 20:05:10 2021

@author: bfildier
"""


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from datetime import datetime as dt
import pytz
import sys,os,glob
import argparse
import pickle
from math import ceil,e,pi


##-- directories and modules

# workdir = '/Users/bfildier/Code/analyses/EUREC4A/EUREC4A_organization/scripts'
workdir = os.path.dirname(os.path.realpath(__file__))
repodir = os.path.dirname(workdir)
moduledir = os.path.join(repodir,'functions')
subdirname = 'radiative_features'
resultdir = os.path.join(repodir,'results',subdirname)
figdir = os.path.join(repodir,'figures',subdirname)
#inputdir = '/Users/bfildier/Dropbox/Data/EUREC4A/sondes_radiative_profiles/'
inputdir = os.path.join(repodir,'input')

# Load own module
projectname = 'EUREC4A_organization'
thismodule = sys.modules[__name__]

## Own modules

sys.path.insert(0,moduledir)
print("Own modules available:", [os.path.splitext(os.path.basename(x))[0]
                                 for x in glob.glob(os.path.join(moduledir,'*.py'))])

from radiativefeatures import *
from radiativescaling import *
from matrixoperators import *

##--- local functions

def defineSimDirectories(day):
        
    # create output directory if not there
    os.makedirs(os.path.join(figdir,day),exist_ok=True)


if __name__ == "__main__":
    
    # arguments
    parser = argparse.ArgumentParser(description="Compute and store features from radiative profile data")
    parser.add_argument('--day', default='20200126',help="day of analysis")
    parser.add_argument('--overwrite',type=bool,nargs='?',default=False)
    
    args = parser.parse_args()
    day = args.day
    
    # day = '20200126'
    date = pytz.utc.localize(dt.strptime(day,'%Y%m%d'))
    
    # # varids
    # ref_varid = 'PW'
    # cond_varids = 'QRAD','QRADSW','QRADLW','QV'
    
    # paths
    defineSimDirectories(day)
    
    mo = MatrixOperators()
    
    ###--- Load data ---###
    
    # Profiles
    radprf = xr.open_dataset(os.path.join(inputdir,'rad_profiles_CF.nc'))
    # choose profiles for that day that start at bottom
    data_all = radprf.where(radprf.z_min<=50,drop=True)
    data_day = data_all.sel(launch_time=day)
    
    # Radiative features
    features_filename = 'rad_features.pickle'
    print('loading %s'%features_filename)
    features_path = os.path.join(resultdir,day,features_filename)
    f = pickle.load(open(features_path,'rb'))
    
    pres = data_day.pressure.data # Pa
    pres_mean = np.nanmean(pres,axis=0)
    temp = data_day.temperature.data
    qv = data_day.specific_humidity.data # kg/kg
    # z = data_day.alt/1e3 # km
    
    
    ##-- compute radiative scaling
    
    print("- compute radiative features")
    
    #---------------------------------------------------------------
    # TO-DO:
    #
    # 1. Replace FeaturesFromXarray() with Features() in computeFatures.py -- ok
    # 2. Rerun analysis_all_day.sh for computeFeatures.py -- ok
    # 3. Replace below with RadiativeScaling() calculation
    # 4. Test for 20210126 and plot a test figure
    # 5. Run analysis_all_day.sh for computeRadiativeScaling.py (current script)
    # 6. In paperFigures.py: . load RadiativeFeatures for all days,
    #                        . show scatter for peak height and peak magintude, all days combined
    #---------------------------------------------------------------

    # Initialize
    rs = RadiativeScaling(pres,temp,qv,f)
    # Compute all terms of radiative scaling (to see detail, look at the method itself)
    rs.computeScaling()
    
    # Save
    rad_scaling_filename = 'rad_scaling.pickle'
    print('saving %s'%rad_scaling_filename)
    rad_scaling_path = os.path.join(resultdir,day,rad_scaling_filename)
    pickle.dump(rs,open(rad_scaling_path,'wb'))
    
#%% ##-- test calculation, graphically
    
    
    

    sys.exit(0)
