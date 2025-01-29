# -*- coding: utf-8 -*-
"""
Jessica Gurney
1/29/2025

Code to get more in depth detail about individual particles using the ReadFile
which looks at txt files like the MW_000.txt
"""

import numpy as np
import astropy.units as u
from ReadFile import Read

def ParticleInfo(filename, p_type, p_num):
    '''
    Extracts distance, velocity magnitude, and mass of a specific particle type and index from file
    parameters:
        filename: str
        Name of the file to read data from
        
        p_type: int
        Particle type (1 = Dark Matter, 2 = Disk Stars, 3 = Bulge Stars)
        
        p_num: int
        Index of the particle (start from 0)
    
    Returns:
        tuple
        (Distance in kpc, velocity in km/s, mass in solar masses)
    '''
    
    #read in the file with ReadFile code
    time, tot_part, data = Read(filename)
    
    #use np.where to find data of specified particles type
    index = np.where(data['type'] == p_type)[0]
    
    #breaking out particle data 
    p_index = index[p_num]                          #takes user index specified and finds below values
    x = data['x'][p_index] * u.kpc                  #x coordinate in kpc
    y = data['y'][p_index] * u.kpc                  #y coordinate in kpc
    z = data['z'][p_index] * u.kpc                  #z coordinate in kpc
    vx = data['vx'][p_index] * u.km / u.s           #velocity of x in km/s
    vy = data['vy'][p_index] * u.km / u.s           #velocity of y in km/s
    vz = data['vz'][p_index] * u.km / u.s           #velocity of z in km/s
    mass = data['m'][p_index] * 1e10 * u.Msun       #converts mass of 10^10 Msun to 1Msun
    
    #using particle data to find 3D distance and magnitudes
    distance = np.sqrt(x**2 + y**2 + z**2)
    velocity = np.sqrt(vx**2 + vy**2 + vz**2)
    
    #convert distance to lightyears from kpc
    dist_ly = distance.to(u.lyr)
    
    #make sure all the values are rounded to first three decimal places
    distance = np.around(distance.value, 3)
    dist_ly = np.around(dist_ly.value, 3)
    velocity = np.around(velocity.value, 3)
    mass = np.around(mass.value,3)
    
    print(f"The Following info for particle type: {p_type}, and index number: {p_num}")
    #print(x, y, z)
    print(f"3D Distance: {distance} kpc")
    print(f"3D Velocity: {velocity} km/s")
    print(f"Mass: {mass} Msun")
    print(f"Distance in Lightyears: {dist_ly} ly")
    
    return distance, velocity, mass

#ParticleInfo('MW_000.txt', p_type=3, p_num=2)