# -*- coding: utf-8 -*-
"""
Jessica Gurney
1/29/2025

Code to use with large data file sets in same format as MW_000.txt
"""

import numpy as np
import astropy.units as u


def Read(filename):
    file = open(filename, 'r')          #open file and read it 
    line1 = file.readline()             #read the first line
    label, value = line1.split()        #split the two columns into label and value
    time = float(value)*u.Myr           #give time a unit of Myr
    
    line2 = file.readline()             #read line two
    label1, value1 = line2.split()      #split two columns into label and value
    tot_part = float(value1)            #grabs the total particles value
    
    file.close()                        #close the damn file
    
    data = np.genfromtxt(filename,dtype=None,names=True,skip_header=3)  #create an array for the rest of the data
    
    #print(data['z'][0])
    
    return time, tot_part, data


#Read('MW_000.txt')