# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:28:44 2018

@author: cana5
"""
# Import the following programs:
from __future__ import division
import tupak
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate

# Define the following to convert values to seconds:
seconds_in_year = 86400 * 365
seconds_in_day = 86400

# Set up some labels for the file and the directory it is saved in:
label = 'PSR-B1828-11_switching_constant_period'
outdir = 'outdir_switching_constant_period'

# Read in the data - this requires you to save the data in /data:
file_name = 'data/1828-11_100vf_fig.dat'
cols = (0, 3, 4)
names = ["MJD", "F1", "F1_err"]
df = pd.read_csv(file_name, sep=' ', usecols=cols, header=None, names=names,
                 dtype=None, skipinitialspace=True, index_col=False)

# This is the independent variable and the dependent variable:
MJD = df.MJD.values
MJD_seconds = MJD * seconds_in_day
nudot = df.F1.values * 1e-15

# Plot the data: 
fig = plt.figure(figsize=(10,6))
plt.plot(MJD_seconds, nudot, label='PRECESSION DATA')
plt.title('PRECESSION DATA FOR PULSAR B1828-11')
plt.legend()
plt.xlabel('TIME (s)')
plt.ylabel('$\dot\\nu$ (Hz/s)')
txt = "Graph of the data points of the pulsar shown as a line graph."
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', 
            fontsize=10)
plt.tight_layout()
plt.savefig('precession_data.pdf')

# Fill in the function to describe precession:
def SignalModel(time, nudot1, nudot2, nuddot, T, tAB, tBC, tCD, phi0, sigma):

    # This is the independent variable minus the MJD value:
    time = time - 49621 * 86400
    
    # This is the number of fine measurements:
    Nfine = 10000
    
    # This is the amount of smoothing
    DeltaT = 100.0*86400  
    
    # This is the fraction to shift by:
    frac_rec = 4  

    # Get the Perera switching of spindown:
    # Define an array:
    time_fine = np.linspace(time[0]-DeltaT/2., time[-1]+DeltaT/2., Nfine)
    
    # This gives an array of zeroes; add the constant 'nudot1'
    F1 = np.zeros(len(time_fine)) + nudot1
    
    # Take the time and create a loop:
    ti_mod = np.mod(time_fine + phi0*T, T)
    F1[(tAB < ti_mod) & (ti_mod < tAB+tBC)] = nudot2
    F1[tAB + tBC + tCD < ti_mod] = nudot2
    F1 = F1 + (nuddot * time_fine)

    # These are the constraints:
    if tAB < tCD or np.abs(nudot1) > np.abs(nudot2):
        F1 = np.zeros(len(F1)) 

    # Integrate to phase:
    F0 = integrate.cumtrapz(y=F1, x=time_fine, initial=0)
    P0 = 2 * np.pi * integrate.cumtrapz(y=F0, x=time_fine, initial=0)

    # Get the average spin-down from the phase:
    dt = time_fine[1] - time_fine[0]
    DeltaT_idx = int(DeltaT / dt)
    
    # Make it even:
    DeltaT_idx += DeltaT_idx % 2  

    tref = time_fine[0]
    time_fineprime = time_fine - tref

    time = time.reshape(len(time), 1)
    deltas = np.abs(time - time_fine)
    idxs = np.argmin(deltas, axis=1)

    vert_idx_list = idxs
    hori_idx_list = np.arange(-DeltaT_idx/2, DeltaT_idx/2, DeltaT_idx/frac_rec, 
                              dtype=int)
    A, B = np.meshgrid(hori_idx_list, vert_idx_list)
    idx_array = A + B

    time_fineprime_array = time_fineprime[idx_array]

    P0_array = P0[idx_array]

    F1_ave = np.polynomial.polynomial.polyfit(time_fineprime_array[0], 
                                              P0_array.T, 2)[2, :]/np.pi

    return F1_ave
   

# This is the 'likelihood' function:
likelihood = tupak.core.likelihood.GaussianLikelihood(x=MJD_seconds, y=nudot, 
                                                      function=SignalModel)

# Fill in the priors/parameters for the appropriate values:
priors = {}

# These values do not have parameters:
priors['nudot1'] = -3.6489 * 10**(-13)
priors['nudot2'] = -3.6635 * 10**(-13)
priors['nuddot'] = 8.75 * 10**(-25)
priors['phi0'] = 0.5278

# Define the above fixed priors explicitly:
fixed_priors = priors.copy()

# These values have a minimum and maximum parameter:
priors['tAB'] = tupak.prior.Uniform(minimum=0, maximum=250 * seconds_in_day, 
      name='tAB')
priors['tBC'] = tupak.prior.Uniform(minimum=70 * seconds_in_day, maximum=250 * 
      seconds_in_day, name='tBC')
priors['tCD'] = tupak.prior.Uniform(minimum=0, maximum=250 * seconds_in_day, 
      name='tCD')
priors['T'] = tupak.prior.Uniform(minimum=450 * seconds_in_day, 
      maximum=550 * seconds_in_day, name='T')

priors['sigma'] = tupak.core.prior.Uniform(0, 1e-15, 'sigma')

# Run the sampler:
result = tupak.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints= 100,
    walks=10, outdir=outdir, label=label, clean=True)
result.plot_corner()