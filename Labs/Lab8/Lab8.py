
# # Lab 8 : Star Formation 

import numpy as np
from astropy import units as u
from astropy import constants as const

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
get_ipython().run_line_magic('matplotlib', 'inline')


# # Part A
# 
# Create a function that returns the SFR for a given luminosity (NUV, FUV, TIR, Halpha)
# 
# $Log( {\rm SFR} (M_\odot/year)) = Log(Lx (erg/s)) - Log(Cx)$ 
# 
# Including corrections for dust absorption 
# 
# Kennicutt & Evans 2012 ARA&A Equation 12 and Table 1, 2

def StarFormationRate(L, Type, TIR=0):
  """
  Function that computes the star formation rate of a galaxy following 
  Kennicutt & Evans 2012  Eq 12 (ARA&A 50)

  Parameters
  __________
    L: 'float'
        luminosity of the galaxy in ergs/s
    Type: 'string'
        The wavelength: 'FUV', 'NUV', 'TIR', 'Halpha'
    TIR: 'float'
        Total infrared luminosity in erg/s (default =0)

  OUTPUT
  ________
    SFR: 'float'
        Log of the Star Formation Rate (Msun/yr)
  """
  if (Type=='FUV'):
    logCx = 43.35 #Calibration from LFUV to SFR
    TIRc = 0.46 #Correction for dust absorption
  elif (Type =='NUV'):
    logCx = 43.17
    TIRc = 0.27
  elif (Type=='Halpha'):
    logCx = 41.27
    TIRc = 0.0024
  elif (Type == 'TIR'):
    logCx = 43.41
    TIRc = 0
  else:
    print("WARNING: Missing the Wavelength. 
          Expecting FUV, NUV, Halpha, TIR.")

  #Correct Luminosity for dust using the TIR
  Lcorr = L + TIRc*TIR

  #star formation rate
  SFR = np.log10(Lcorr) - logCx

  return SFR
  

# Let's try to reproduce SFRs derived for the WLM Dwarf Irregular Galaxy using UV luminosities measured with Galex. 
# 
# Compare results to Table 1 from Lee et al. 2009 (who used the older Kennicutt 98 methods)
# https://ui.adsabs.harvard.edu/abs/2009ApJ...706..599L/abstract
# 
# We will use galaxy properties from NED (Photometry and SED):
# https://ned.ipac.caltech.edu/


# First need the Luminosity of the Sun in the right units
const.L_sun 
LsunErgs = const.L_sun.to(u.erg/u.s).value
print (LsunErgs)

#  WLM Dwarf Irregular Galaxy
# From Ned Galex Data
NUV_WLM = 1.71e7*LsunErgs
TIR_WLM = 2.48e6*LsunErgs + 3.21e5*LsunErgs + 2.49e6*LsunErgs
#TIR = NIR + MIR + FIR

#Test
StarFormationRate(1e6*LsunErgs, 'blah')
StarFormationRate(NUV_WLM, 'NUV', TIR_WLM)


# # Part B Star formation main sequence
# 
# 1) Write a function that returns the average SFR of a galaxy at a given redshift, given its stellar mass
# 
# 2) What is the average SFR of a MW mass galaxy today? at z=1?
# 
# 3) Plot the SFR main sequence for a few different redshifts from 1e9 to 1e12 Msun.
# 
# 
# From Whitaker 2012:
# 
# log(SFR) = $\alpha(z)({\rm log}M_\ast - 10.5) + \beta(z)$
# 
# $\alpha(z) = 0.7 - 0.13z$
# 
# $\beta(z) = 0.38 + 1.14z - 0.19z^2$

# # Step 1

def SFRMainSequence(Mstar,z):
  """
  Function that computes the average SFR of a galaxy
  as a function of stellar mass and redshift.
  PARAMETERS
  ----------
    Mstar: 'float'
      Stellar mass of the galaxy in Msun
    z: 'float'
      Redshift
  OUTPUT
  ------
    SFR:'float'
      Log of the Star Formation Rate SFR (Msun/yr)
  """
  alpha = 0.7-0.13*z
  beta = 0.38 + 1.14*z - 0.19*z**2

  SFR = alpha*(np.log10(Mstar)-10.5)+beta
  return SFR


# # Step 2



# MW at z=0




# MW at z = 1


# # Step 3



# create an array of stellar masses





fig = plt.figure(figsize=(8,8), dpi=500)
ax = plt.subplot(111)

# add log log plots


# Add axis labels
plt.xlabel('Log(Mstar (M$_\odot$))', fontsize=12)
plt.ylabel('Log(SFR (M$_\odot$/year))', fontsize=12)


#adjust tick label font size
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='upper left',fontsize='x-large')


# Save file
#plt.savefig('Lab8_SFR_MainSequence.png')


# # Part C  Starbursts
# 
# Use your `StarFormationRate` code to determine the typical star formation rates for the following systems with the listed Total Infrared Luminosities (TIR): 
# 
# Normal Galaxies: $10^{10}$ L$_\odot$
# 
# LIRG: $10^{11}$ L$_\odot$
# 
# ULIRG: $10^{12} $ L$_\odot$
# 
# HLIRG: $10^{13} $ L$_\odot$



# normal galaxies 




# LIRGs  




# ULIRGs




# HLIRGs






