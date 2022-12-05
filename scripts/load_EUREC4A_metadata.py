#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EUREC4A metadata

Created on Tue Aug  9 15:22:17 2022

@author: bfildier
"""

import sys, os, glob
import numpy as np

#-- EUREC4A metadata


days =          '20200122','20200124','20200126','20200128','20200131','20200202','20200205','20200207','20200209','20200211','20200213'
name_pattern =  'Fish',    'Fish',    'Fish',    'Gravel',  'Fish',    'Flower',  'Gravel',  'Flower',    'Sugar',   'Sugar',   'Fish'
confidence_pattern = 'High','Medium', 'Medium',     'Low',     'Low',     'High',    'High',    'High',  'Medium',  'Medium',  'High'
col_pattern = {'':'silver',
               'Fish':'navy',
               'Gravel':'orange',
               'Sugar':'seagreen',
               'Flower':'firebrick'}

dim_t,dim_z = 0,1

# box GOES images
lat_box_goes = 10,16
lon_box_goes = -60,-52

# box of analysis
lat_box = 11,16
lon_box = -60,-52

# HALO circle coordinates
lon_center = -57.717
lat_center = 13.3
lon_pt_circle = -57.245
lat_pt_circle = 14.1903
r_circle = np.sqrt((lon_pt_circle - lon_center) ** 2 +
                   (lat_pt_circle - lat_center) ** 2)

# varids
ref_varid = 'PW'
cond_varids = 'QRAD','QRADSW','QRADLW','QV','UNORM','T','P'

# high peaks
days_high_peaks = '20200213', '20200211', '20200209','20200128', '20200124', '20200122'
z_min_all = 4, 4, 3.5, 4, 3, 3.2
z_max_all = 9, 6, 8.5, 6, 4.5, 4

# intrusions!
days_intrusions = '20200213', '20200213', '20200211', '20200209', '20200209', '20200128', 
z_min_intrusions =  4, 5, 4, 3, 5.2, 3.5 
z_max_intrusions =  5, 9, 6, 5.2, 8, 6