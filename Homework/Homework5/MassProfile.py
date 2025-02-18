# -*- coding: utf-8 -*-
"""
Jessica Gurney
HW5 Due: Feb 20, 2025
"""

import numpy as np
import astropy.units as u
from astropy.constants import G
from ReadFile import Read  
from CenterOfMass import CenterOfMass as COM  
import matplotlib.pyplot as plt

class MassProfile:
    def __init__(self, galaxy, snap):
        """
        Initialize the MassProfile class. Example if put in "MW" and "10" 
        should output MW_010.txt
        
        Parameters:
        galaxy : str
            Name of the galaxy (e.g., "MW", "M31", "M33")
        snap : int
            Snapshot number
        """
        # Add a string of the filenumber to the value "000"
        ilbl = '000' + str(snap)
        ilbl = ilbl[-3:]  # remove all but the last 3 digits
        self.filename = f"{galaxy}_{ilbl}.txt"
        
        # Store galaxy name as a class property
        self.gname = galaxy
        
        # Read data from file
        self.time, self.total, self.data = Read(self.filename)
        
        # Extract positions and masses
        self.x = self.data['x'] * u.kpc  # Convert to kpc
        self.y = self.data['y'] * u.kpc
        self.z = self.data['z'] * u.kpc
        self.m = self.data['m']  # Mass will be converted later

    def MassEnclosed(self, ptype, radii):
        """
        Calculate the amount of mass inside a specified radius from a chosen
        galaxy and galaxy component type
        
        Parameters:
        ptype : int
            Particle type (1 = Halo, 2 = Disk, 3 = Bulge)
        radii : array
            Array of radii distances in magnitude (kpc)
        
        Returns:
        np.array
            Enclosed mass at each radius (Msun)
        """
        
        # Determine the COM position 
        com = COM(self.filename, 2)  #COM is based on Disk Particles only, so default 2
        com_p = com.COM_P(0.1)  #3D vector of COM with error tolerance default of 0.1 kpc
        
        # Select particles of the specified type of 
        index = np.where(self.data['type'] == ptype)
        x_p = self.x[index] - com_p[0]   #subtract center of mass of x, y, z, distance
        y_p = self.y[index] - com_p[1]   #to shift all particles so COM is at origin
        z_p = self.z[index] - com_p[2]
        m_p = self.m[index]
        
        # Compute radial distance from COM
        r_p = np.sqrt(x_p**2 + y_p**2 + z_p**2)
        
        # Initialize array to store enclosed masses
        enclosed_mass = np.zeros(len(radii))
        
        # Loop over radii and sum masses within each radius
        for i in range(len(radii)):
            enclosed_mass[i] = np.sum(m_p[r_p < radii[i]])
        
        return enclosed_mass * u.Msun

    def MassEnclosedTotal(self, radii):
        """
        Compute the total enclosed mass (halo + disk + bulge) within a given radius.
        
        Parameters:
        radii : array
            Array of radii 1D (kpc)
        
        Returns:
        np.array
            Total enclosed mass at each radius (Msun)
        """
        halo_mass = self.MassEnclosed(1, radii)  #get halo mass
        disk_mass = self.MassEnclosed(2, radii)  #get disk mass
        
        if self.gname == "M33":
            bulge_mass = np.zeros(len(radii)) * u.Msun  # For M33 creates bulge as zero for all radii
        else:
            bulge_mass = self.MassEnclosed(3, radii)  #get bulge mass
        
        return halo_mass + disk_mass + bulge_mass

    def HernquistMass(self, r, a, Mhalo):
        """
        Compute the mass enclosed within a given radius using the Hernquist profile.
        
        Parameters:
        r : float or array
            Radius (kpc)
        a : float
            Scale factor (kpc)
        Mhalo : float
            Total halo mass (Msun)
        
        Returns:
        float or array
            Hernquist enclosed mass (Msun)
        """
        # given formula for HernquistMass
        # M(r) = (Mhalo*r^2)/(a+r)^2
        r = r.to(u.kpc)
        a = a * u.kpc
        Mhalo = Mhalo * u.Msun
        
        Hmass = (Mhalo * r**2) / ((a + r)**2)
        
        return Hmass * u.Msun
  
    def CircularVelocity(self, ptype, radii):
        """
        Compute the circular velocity at a given radius.
        
        Parameters:
        ptype : int
            Particle type (1 = Halo, 2 = Disk, 3 = Bulge)
        radii : array
            Array of radii (kpc)
        
        Returns:
        np.array
            Circular velocity at each radius (km/s)
        """
        #Circular velocity of a specific type of particle in a galaxy
        
        # Convert gravitational constant to suitable units to get right velocity units later
        G_converted = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun)  
        
        # Compute mass enclosed at each radius assuming spherical symmetry
        M_enc = self.MassEnclosed(ptype, radii)
        
        # Compute circular velocity
        # Vcirc = sqrt(G*M/r)
        V_circ = np.sqrt(G_converted * M_enc / radii).to(u.km/u.s) #return in units of km/s
        
        return np.round(V_circ, 2)  # Round to two decimal places
    
    def CircularVelocityTotal(self, radii):
        """
        Compute the total circular velocity considering all components (halo, disk, bulge).
        
        Parameters:
        radii : array
            Array of radii (kpc)
        
        Returns:
        np.array
            Total circular velocity at each radius (km/s)
        """
        #Take total enclosed mass (MassEnclosedTotal) and apply circular velocity
        #to total enclosed mass. Repeat function from above but using total mass. 
        
        G_converted = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun)
        M_enc_total = self.MassEnclosedTotal(radii)
        
        # Vcirc = sqrt(G*M/r)        
        V_circ_total = np.sqrt(G_converted * M_enc_total / radii).to(u.km/u.s)
        
        return np.round(V_circ_total, 2)
    
    def HernquistVCirc(self, r, a, Mhalo):
        """
        Compute the circular velocity using the Hernquist mass profile.
        
        Parameters:
        r : float or array
            Radius (kpc)
        a : float
            Scale factor (kpc)
        Mhalo : float
            Total halo mass (Msun)
        
        Returns:
        float or array
            Hernquist circular velocity (km/s)
        """
        #same functions as above, but using the Hernquist mass in the formula
        G_converted = G.to(u.kpc * u.km**2 / u.s**2 / u.Msun)
        M_enc = self.HernquistMass(r, a, Mhalo)
        
        # Vcirc = sqrt(G*M/r)
        V_circ = np.sqrt(G_converted * M_enc / r).to(u.km/u.s)
        
        return np.round(V_circ, 2)
    
    def plot_mass_profile(self, radii, a_hernquist=None, Mhalo=None):
        """
        Plot the mass profile for each component of the galaxy.

        Parameters:
        radii : array
            Array of radii (kpc)
        a_hernquist : float, optional
            Best-fit scale radius for the Hernquist profile (default is None).
        Mhalo : float, optional
            Halo mass (Msun) for the Hernquist profile (default is None).
        """
        # Compute mass for each component
        halo_mass = self.MassEnclosed(1, radii)
        disk_mass = self.MassEnclosed(2, radii)
        bulge_mass = self.MassEnclosed(3, radii) if self.gname != "M33" else np.zeros(len(radii)) * u.Msun
        total_mass = self.MassEnclosedTotal(radii)

        # Plot the components with different colors and line styles
        plt.figure(figsize=(8, 6))
        plt.semilogy(radii, halo_mass, label='Halo', color='b', linestyle='--')
        plt.semilogy(radii, disk_mass, label='Disk', color='g', linestyle='-.')
        plt.semilogy(radii, bulge_mass, label='Bulge', color='r', linestyle=':')
        plt.semilogy(radii, total_mass, label='Total', color='k', linewidth=2)

        # If Hernquist profile is provided, plot it
        if a_hernquist is not None and Mhalo is not None:
            hernquist_mass = self.HernquistMass(radii, a_hernquist, Mhalo)
            plt.plot(radii, hernquist_mass, label=f'Hernquist Fit (a={a_hernquist:.2f} kpc)', color='purple', linestyle='--')

        # Label the axes and the plot
        plt.xlabel('Radius (kpc)')
        plt.ylabel('Mass Enclosed (Msun)')
        plt.title(f'Mass Profile for {self.gname}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
galaxy = 'MW'  # For Milky Way, or use 'M31' or 'M33'
snap = 0      # Snapshot number
mass_profile = MassProfile(galaxy, snap)

# Define radii (start at 0.1 kpc, go up to 30 kpc)
radii = np.linspace(0.1, 30, 100) * u.kpc

# Call the plotting method (adjust 'a' and 'Mhalo' for Hernquist fit if needed)
mass_profile.plot_mass_profile(radii, a_hernquist=15, Mhalo=1e12)  # Example Hernquist fit params    































































'''
def plot_mass_profile(galaxy, snap, Mhalo, a_guess):
    """
    Plot the mass enclosed as a function of radius for different galaxy components.
    
    Parameters:
    galaxy : str
        Name of the galaxy (MW, M31, M33)
    snap : int
        Snapshot number
    Mhalo : float
        Total halo mass in Msun
    a_guess : float
        Initial guess for the Hernquist scale radius (kpc)
    """
    # Initialize the MassProfile class
    mp = MassProfile(galaxy, snap)
    
    # Define an array of radii from 0.1 to 30 kpc using 100 steps
    radii = np.linspace(0.1, 30, 100) 
    
    # Compute mass profiles
    halo_mass = mp.MassEnclosed(1, radii)
    disk_mass = mp.MassEnclosed(2, radii)
    bulge_mass = np.zeros(len(radii)) * u.Msun if galaxy == "M33" else mp.MassEnclosed(3, radii)
    total_mass = mp.MassEnclosedTotal(radii)
    
    # Find the best-fitting Hernquist scale radius (adjust manually)
    hernquist_mass = mp.HernquistMass(radii, a_guess, Mhalo)
    
    # Plot mass profiles
    plt.figure(figsize=(8, 6))
    plt.semilogy(radii, halo_mass, label="Halo", linestyle='dashed', color='blue')
    plt.semilogy(radii, disk_mass, label="Disk", linestyle='dotted', color='red')
    if galaxy != "M33":
        plt.semilogy(radii, bulge_mass, label="Bulge", linestyle='dashdot', color='green')
    plt.semilogy(radii, total_mass, label="Total", linestyle='solid', color='black')
    plt.semilogy(radii, hernquist_mass, label=f"Hernquist (a={a_guess} kpc)", linestyle='dashed', color='purple')
    
    # Labels and legend
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Mass Enclosed (Msun)")
    plt.title(f"{galaxy} Mass Profile")
    plt.legend()
    plt.grid()
    plt.show()

plot_mass_profile(galaxy='MW', snap=0, Mhalo, a_guess=)





def plot_rotation_curve(galaxy, snap, Mhalo, a_guess):
    """
    Plot the rotation curve (circular velocity) as a function of radius for different galaxy components.
    
    Parameters:
    galaxy : str
        Name of the galaxy (MW, M31, M33)
    snap : int
        Snapshot number
    Mhalo : float
        Total halo mass in Msun
    a_guess : float
        Initial guess for the Hernquist scale radius (kpc)
    """
    # Initialize the MassProfile class
    mp = MassProfile(galaxy, snap)
    
    # Define an array of radii from 0.1 to 30 kpc (avoid starting from zero)
    radii = np.arange(0.1, 30.5, 0.5) * u.kpc
    
    # Compute circular velocities for each component
    halo_vcirc = mp.CircularVelocity(1, radii)
    disk_vcirc = mp.CircularVelocity(2, radii)
    bulge_vcirc = np.zeros(len(radii)) * u.km/u.s if galaxy == "M33" else mp.CircularVelocity(3, radii)
    total_vcirc = mp.CircularVelocityTotal(radii)
    
    # Compute the best-fit Hernquist circular velocity
    hernquist_vcirc = mp.HernquistVCirc(radii, a_guess, Mhalo)
    
    # Plot the rotation curves for each component
    plt.figure(figsize=(8, 6))
    plt.plot(radii, halo_vcirc, label="Halo", linestyle='dashed', color='blue')
    plt.plot(radii, disk_vcirc, label="Disk", linestyle='dotted', color='red')
    if galaxy != "M33":
        plt.plot(radii, bulge_vcirc, label="Bulge", linestyle='dashdot', color='green')
    plt.plot(radii, total_vcirc, label="Total", linestyle='solid', color='black')
    plt.plot(radii, hernquist_vcirc, label=f"Hernquist (a={a_guess} kpc)", linestyle='dashed', color='purple')
    
    # Labels and legend
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Circular Velocity (km/s)")
    plt.title(f"{galaxy} Rotation Curve")
    plt.legend()
    plt.grid()
    plt.show()

# Example usage:
plot_rotation_curve("MW", 0, Mhalo=1.5e12, a_guess=20)
plot_rotation_curve("M31", 0, Mhalo=2.5e12, a_guess=25)
plot_rotation_curve("M33", 0, Mhalo=2.0e11, a_guess=15)
'''
