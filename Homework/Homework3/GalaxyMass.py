# -*- coding: utf-8 -*-
"""
Jessica Gurney
Due: 2_6_2025

"""

from ReadFile import Read
from astropy import units as u
from astropy.table import Table
import numpy as np

def ComponentMass(file, part_type):
    """ 
    This function will return the total mass of desired galaxy component. 
    Using program ReadFile
  
    Inputs: file (str) name of file containing info on galaxy
              hint assumes mass is in 1e10 Msun in file
         
          part_type (num) Particle type of galaxy
              1 (Halo Type)
              2 (Disk Type)
              3 (Bulge Type)
  
    Outputs: (float) 
      Returns the total mass (10^12 Msun) of any desired galaxy component 
  """
    time,tot_part,data = Read(file)  #use Read function to pull galaxy info from file
    
    #sum up particles of given type
    mass = np.sum(data['m'][data['type'] == part_type]) * 1e10 * u.Msun
    
    #convert to 1e12 Msun by first converting to Msun, then dividing to get 1e12, round 3 decimal
    tot_mass = np.round(mass.to(u.Msun)/1e12,3)*u.Msun
        
    return tot_mass

  
#Compute mass for each galaxy
galaxies = ['MW', 'M31', 'M33']
files = ['MW_000.txt', 'M31_000.txt', 'M33_000.txt']

data_for_table = [] #store the data collected 

#Loop through galaxies to get mass of components, total mass, and baryon fraction
for galaxy, file in zip(galaxies, files):
    galaxy_data = {} #create dictionary for storing data
    
    #compute mass of each component, assign to dictionary term, example Halo Mass = 14*1e12 Msun
    galaxy_data['Galaxy Name'] = galaxy
    galaxy_data['Halo Mass'] = ComponentMass(file,1) # 1=Halo type
    galaxy_data['Disk Mass'] = ComponentMass(file,2) # 2=Disk Type
    galaxy_data['Bulge Mass'] = ComponentMass(file,3) # 2=Bulge Type
    
    #Total mass of full galaxy, add the values of each dictionary 
    total_mass = galaxy_data['Halo Mass'] + galaxy_data['Disk Mass'] + galaxy_data['Bulge Mass']
    #create new dictionary term for total mass
    galaxy_data['Total Mass'] = total_mass
    
    #baryon fraction = total stella mass/ total mass(dark+stellar) 
    #Dark mass = Halo
    #Stellar mass = Disk and Bulge
    fbar = (galaxy_data['Disk Mass'] + galaxy_data['Bulge Mass'])/total_mass
    galaxy_data['fbar'] = round(fbar.value,3)
    
    #append galaxy data to data list
    data_for_table.append(galaxy_data)
    
#Crate an astropy table from the data list
galaxy_table = Table(rows=data_for_table, names=('Galaxy Name', 'Halo Mass','Disk Mass',
                                                'Bulge Mass','Total Mass', 'fbar'))

#compute the total mass of the local group and its fbar
tot_lg_mass = np.sum([row['Total Mass'].value for row in data_for_table])
tot_lg_stellar_mass = np.sum([ (row['Disk Mass'] + row['Bulge Mass']).value for row in data_for_table])
fbar_lg = tot_lg_stellar_mass / tot_lg_mass
fbar_lg = round(fbar_lg, 3)

#add local group totals as a new row to the astropy table
galaxy_table.add_row(('Local Group', 0 * u.Msun, 0 * u.Msun, 0 * u.Msun, tot_lg_mass * u.Msun, fbar_lg))

# Assign units to relevant columns
galaxy_table.rename_column('Halo Mass', 'Halo Mass (10^12 Msun)')
galaxy_table.rename_column('Disk Mass', 'Disk Mass (10^12 Msun)')
galaxy_table.rename_column('Bulge Mass', 'Bulge Mass (10^12 Msun)')
galaxy_table.rename_column('Total Mass', 'Total Mass (10^12 Msun)')

#print table
print(galaxy_table)

#save table as pdf
galaxy_table.write("Galaxy_Mass_Breakdown.csv", format='csv', overwrite=True)
    
    

