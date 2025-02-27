'''
Jessica Gurney
HW6
Due 2/27/25
'''

# Homework 6 Template
# G. Besla & R. Li


# import modules
import numpy as np
import astropy.units as u
from astropy.constants import G

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules
from ReadFile import Read
# Step 1: modify CenterOfMass so that COM_P now takes a parameter specifying 
# by how much to decrease RMAX instead of a factor of 2
from CenterOfMass import CenterOfMass



def OrbitCOM(galaxy, start, end, n):
    '''function that loops over all the desired snapshots to compute the COM pos and vel as a function of time.
    inputs: 
        galaxy (str)
        the name of a galaxy "MW"
        
        start (float)
        the number of the first snapshot to be read in
        
        end (float)
        the number of the last snapshot to be read in
        
        n (int)
        an integer indicating the intervals over which you will return the COM
          
    outputs: !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    '''
    
    # compose the filename for output
    fileout = f"Orbit_{galaxy}.txt"

    
    #  set tolerance and VolDec for calculating COM_P in CenterOfMass
    # for M33 that is stripped more, use different values for VolDec
    delta = 0.1
    if galaxy == 'M33':
        volDec = 4 
    else:
        volDec = 2
    
    # generate the snapshot id sequence 
    # it is always a good idea to also check if the input is eligible (not required)
    snap_ids = np.arange(start, end, n)
    if snap_ids.size == 0:
        print("Error: The array is empty. Check start, end, and n values")
    
    # initialize the array for orbital info: t, x, y, z, vx, vy, vz of COM
    orbit = np.zeros([len(snap_ids), 7])  #number of rows same length as snaps, 7 columns for orbital info
    
    # loop computes COM positions and velocities at each snapshot of simulation.
    for i, snap_id in enumerate(snap_ids): # loop over files
        
        # compose the data filename (be careful about the folder)
        ilbl = '000' + str(snap_id)  #add a string of the filenumber to the value '000'
        ilbl = ilbl[-3:]    #remove all but the last 3 digits
        filename = f"{galaxy}/{galaxy}_{ilbl}.txt"
        
        # Initialize an instance of CenterOfMass class, using disk particles
        com = CenterOfMass(filename, ptype=2)

        # Store the COM pos and vel. Remember that now COM_P required VolDec
        com_p = com.COM_P(delta, volDec)
        com_v = com.COM_V(com_p[0],com_p[1],com_p[2])

        # store the time, pos, vel in ith element of the orbit array,  without units (.value) 
        # note that you can store 
        # a[i] = var1, *tuple(array1)
        orbit[i, 0] = com.time.value / 1000.0 # Store time in Gyr
        orbit[i, 1:4] = com_p.value  # Store x, y, z position (in kpc)
        orbit[i, 4:7] = com_v.value  # Store vx, vy, vz velocity (in km/s)

        
        # print snap_id to see the progress
        print(f"Completed snapshot {snap_id}")
        
    # write the data to a file
    # we do this because we don't want to have to repeat this process 
    # this code should only have to be called once per galaxy.
    np.savetxt(fileout, orbit, fmt = "%11.3f"*7, comments='#',
               header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                      .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))




# Recover the orbits and generate the COM files for each galaxy
# read in 800 snapshots in intervals of n=5
# Note: This might take a little while - test your code with a smaller number of snapshots first! 

OrbitCOM(galaxy='MW', start=0, end=800, n=5) 
OrbitCOM(galaxy='M31', start=0, end=800, n=5)
OrbitCOM(galaxy='M33', start=0, end=800, n=5)

# Read in the data files for the orbits of each galaxy that you just created
# headers:  t, x, y, z, vx, vy, vz
# using np.genfromtxt

MW_data = np.genfromtxt("Orbit_MW.txt", comments='#')
M31_data = np.genfromtxt("Orbit_M31.txt", comments='#')
M33_data = np.genfromtxt("Orbit_M33.txt", comments='#')


# function to compute the magnitude of the difference between two vectors 
# You can use this function to return both the relative position and relative velocity for two 
# galaxies over the entire orbit  

def vector_magnitude(vector):
    """Compute the magnitude of a 3D vector.
    Input:
        vector (np.array) : 3D vector [x, y, z] or [vx, vy, vz]
    Output:
        magnitude (float) : Magnitude of the vector
    """
    return np.sqrt(np.sum(vector**2))
 


# Determine the magnitude of the relative position and velocities 
# of MW and M31
pos_MW_M31 = MW_data[:, 1:4] - M31_data[:, 1:4]  # x, y, z 
vel_MW_M31 = MW_data[:, 4:7] - M31_data[:, 4:7]  # vx, vy, vz

# Determine the magnitude of the relative position and velocities
# of M33 and M31
pos_M33_M31 = M33_data[:, 1:4] - M31_data[:, 1:4] # x, y, z
vel_M33_M31 = M33_data[:, 4:7] - M31_data[:, 4:7] # vx, vy, vz


# Magnitudes of Relative Position and Velocity
# M31 and MW
sep_MW_M31 = np.array([vector_magnitude(v) for v in pos_MW_M31])
vel_MW_M31 = np.array([vector_magnitude(v) for v in vel_MW_M31])

# Magnitudes of Relative Position and Velocity
# M33 and M31
sep_M33_M31 = np.array([vector_magnitude(v) for v in pos_M33_M31])
vel_M33_M31 = np.array([vector_magnitude(v) for v in vel_M33_M31])



# Plot the Orbit of the galaxies 
#################################
# Time array (in Gyr)
time = MW_data[:, 0]

# Separation Plot
plt.figure(figsize=(10, 6))
plt.plot(time, sep_MW_M31, label='MW-M31', color='blue')
plt.plot(time, sep_M33_M31, label='M33-M31', color='green')
plt.xlabel('Time (Gyr)')
plt.ylabel('Separation (kpc)')
plt.title('Relative Separation of Galaxies')
plt.legend()
plt.grid(True)
plt.show()

# Separation Plot (log)
plt.figure(figsize=(10, 6))
plt.plot(time, sep_MW_M31, label='MW-M31', color='blue')
plt.plot(time, sep_M33_M31, label='M33-M31', color='green')
plt.yscale('log')
plt.xlabel('Time (Gyr)')
plt.ylabel('Separation (kpc)')
plt.title('Relative Separation of Galaxies (Log)')
plt.legend()
plt.grid(True)
plt.show()


# Plot the orbital velocities of the galaxies 
#################################
# Velocity Plot
plt.figure(figsize=(10, 6))
plt.plot(time, vel_MW_M31, label='MW-M31', color='blue')
plt.plot(time, vel_M33_M31, label='M33-M31', color='green')
plt.xlabel('Time (Gyr)')
plt.ylabel('Relative Velocity (km/s)')
plt.title('Relative Velocity of Galaxies')
plt.legend()
plt.grid(True)
plt.show()