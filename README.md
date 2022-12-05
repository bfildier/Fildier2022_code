### 0. Packages and python environment

Load conda environment by executing 
conda activate pyF22

### 1. Homemade modules (directory $HOME/functions)

- radiativefeatures.py : to calculate peak radiative cooling characteristics and moisture structure characteristics (water paths, beta coefficient)
- radiativescaling.py : to calculate scaling approximations
- matrixoperators.py : for some basic matrix manipulations (vertical integrals, smoothing operators, etc.)
- thermoConstants.py : physical constants
- thermoFunctions.py : thermodynamic functions (e.g. saturated vapor pressure)

### 2. Before running scripts, link large data files to your repo

- input/rad_profiles_moist_intrusions_20200213lower.nc
- input/rad_profiles_moist_intrusions_20200213lower_fix_k.nc
- input/rad_profiles_moist_intrusions_20200213lower_fix_T.nc
- input/rad_profiles_moist_intrusions_20200213lower_fix_k_T.nc
- results/idealized_calculations/kappa_fit.pickle
- input/rad_profiles_CF.nc

### 3. Scripts

Files:
- computeKappaNuFit.py: calculates parameters for exponential fit of extinction coefficient kappa as a function of wavenumber, in the rotational and vibration-rotation bands of water vapor

- analysis_all_days.sh: computes all features from observed radiative cooling (computeFeatures.py) and the analytical scaling approximations (computeRadiativeScaling.py)

- loadPatternSnapshots.sh: downloads GOES images for each day and adds the spatial coordinates, date and HALO circle (showPatternSnapshot.py)

- FigureX.py: load all analysis results (load_data.py) and draws Figure X

- computeNumbers.py calculates all numbers:
    . the most emitting wavenumbers for reference temperature and water vapor paths
    . alpha exponent in the power expression for saturated specific humidity
    . the effet of a temperature inversion
    . free-tropospheric water paths at reference relative humidity values
    . analytical values for Appendix B
