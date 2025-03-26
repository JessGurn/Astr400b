'''
Jessica Gurney
HW7 03_26_2025
'''
# # Homework 7 Template
# 
# Rixin Li & G . Besla
# 
# Make edits where instructed - look for "****", which indicates where you need to 
# add code. 


# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass import CenterOfMass


# **** import the GalaxyMass to determine the mass of M31 for each component
from GalaxyMass import ComponentMass

# # M33AnalyticOrbit




class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self, filename): # **** add inputs
        """ Initialize class attributes
        
        Inputs
        ------
            filename: str
                Name of the file to store integrated orbit data
        """

        ### get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        
        ### **** store the output file name
        self.filename = filename
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        M33_COM = CenterOfMass("M33_000.txt", 2)
        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        self.rM33 = M33_COM.COM_P(0.1)
        # **** store the velocity VECTOR of the M33 COM (.value to get rid of units)
        self.vM33 = M33_COM.COM_V(self.rM33[0], self.rM33[1], self.rM33[2]).value
        
        ### get the current pos/vel of M31 
        # **** create an instance of the  CenterOfMass class for M31 
        M31_COM = CenterOfMass("M31_000.txt", 2)
        # **** store the position VECTOR of the M31 COM (.value to get rid of units)
        self.rM31 = M31_COM.COM_P(0.1)
        # **** store the velocity VECTOR of the M31 COM (.value to get rid of units)
        self.vM31 = M31_COM.COM_V(self.rM31[0], self.rM31[1], self.rM31[2]).value
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33
        self.r0 = np.array(self.rM33) - np.array(self.rM31)
        self.v0 = np.array(self.vM33) - np.array(self.vM31)
        
        
        ### get the mass of each component in M31 
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5 #kpc
        # **** self.Mdisk set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mdisk = ComponentMass("M31_000.txt", 2) * 1e12 #Msun
        ### bulge
        # **** self.rbulge = set scale length (no units)
        self.rbulge = 1 #kpc
        # **** self.Mbulge  set with ComponentMass function. Remember to *1e12 to get the right units Use the right ptype
        self.Mbulge = ComponentMass("M31_000.txt", 3) * 1e12 #Msun
        # Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        self.rhalo = 25 #kpc (from HW5)
        # **** self.Mhalo set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mhalo = 1.921e12 #Msun (from HW3)
    
    
    def HernquistAccel(self, M, ra, r): # it is easiest if you take as an input the position VECTOR 
        """ Compute acceleration due to Hernquist potential
        
        Inputs
        ------
            M: 'float'
                Mass of the component (Msun)
            ra: 'float'
                Scale length (kpc)
            r: 'array'
                Position vector (kpc)
            
        Outputs
        -------
            Hern: 'array'
                Acceleration vector (kpc/Gyr^2)
        """
        
        ### **** Store the magnitude of the position vector
        rmag = np.linalg.norm(r)
        
        ### *** Store the Acceleration
        Hern =  -self.G * M / (rmag * (ra + rmag)**2) * r 
        # follow the formula in the HW instructions
        # NOTE: we want an acceleration VECTOR so you need to make sure that in the Hernquist equation you 
        # use  -G*M/(rmag *(ra + rmag)**2) * r --> where the last r is a VECTOR 
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self, M, rd, r):# it is easiest if you take as an input a position VECTOR  r 
        """ Compute acceleration due to Miyamoto-Nagai potential
        
        Inputs
        ------
            M: 'float'
                Mass of the disk (Msun)
            rd: 'float'
                Scale length (kpc)
            r: 'array'
                Position vector (kpc)
            
        Outputs
        -------
            accel: array
                Acceleration vector (kpc/Gyr^2)
        """
        
        ### Acceleration **** follow the formula in the HW instructions
        # AGAIN note that we want a VECTOR to be returned  (see Hernquist instructions)
        # this can be tricky given that the z component is different than in the x or y directions. 
        # we can deal with this by multiplying the whole thing by an extra array that accounts for the 
        # differences in the z direction:
        #  multiply the whle thing by :   np.array([1,1,ZSTUFF]) 
        # where ZSTUFF are the terms associated with the z direction
        
        R = np.sqrt(r[0]**2 + r[1]**2)
        zd = rd/5.0
        B = rd + np.sqrt(r[2]**2 + zd**2)
        accel = -self.G * M / (R**2 + B**2)**1.5 * r * np.array([1, 1, B / np.sqrt(r[2]**2 + zd**2)])
       
        return accel
        # the np.array allows for a different value for the z component of the acceleration
     
    
    def M31Accel(self, r): # input should include the position vector, r
        """ Compute total acceleration from M31's halo, bulge, and disk
        
            Inputs
            ------
                r: 'array'
                    Position vector (kpc)
                
            Outputs
            -------
                total_accel: array
                    Total acceleration vector (kpc/Gyr^2)
        """

        ### Call the previous functions for the halo, bulge and disk
        # **** these functions will take as inputs variable we defined in the initialization of the class like 
        # self.rdisk etc. 
        ahalo = self.HernquistAccel(self.Mhalo, self.rhalo, r)
        abulge = self.HernquistAccel(self.Mbulge, self.rbulge, r)   
        adisk = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)         
            # return the SUM of the output of the acceleration functions - this will return a VECTOR 
        return ahalo + abulge + adisk
    
    
    
    def LeapFrog(self, r, v, dt): # take as input r and v, which are VECTORS. Assume it is ONE vector at a time
        """ Perform a single Leapfrog Integration step
        
            Inputs
            ------
                r: 'array'
                    Current position vector (kpc)
                v: 'array'
                    Current velocity vector (kpc/Gry)
                dt: 'float'
                    Time step (Gyr)
                
            Outputs
            -------
                rnew: 'array'
                    Updated position vector (kpc)
                vnew: 'array'
                    Updated velocity vector (kpc/Gyr)
        """
        
        # predict the position at the next half timestep
        rhalf = r + v * (dt/2)
        
        # predict the final velocity at the next timestep using the acceleration field at the rhalf position 
        vnew = v + self.M31Accel(rhalf) * dt
        
        # predict the final position using the average of the current velocity and the final velocity
        # this accounts for the fact that we don't know how the speed changes from the current timestep to the 
        # next, so we approximate it using the average expected speed over the time interval dt. 
        rnew = rhalf + vnew * (dt/2)
        
        return rnew, vnew # **** return the new position and velcoity vectors
    
    
    
    def OrbitIntegration(self, t0, dt, tmax):
        """ Integrate the orbit using Leapfrog method
        
        Inputs
        ------
            t0: 'float'
                Initial time (Gyr)
            dt: 'float'
                Time step (Gyr)
            tmax: 'float'
                Maximum time (Gyr)
            
        Outputs
        -------
            None (writes orbite data to file)
        """

        # initialize the time to the input starting time
        t = t0
        
        # initialize an empty array of size :  rows int(tmax/dt)+2  , columns 7
        orbit = np.zeros((int((tmax-t0)/dt) + 2,7))
        
        # initialize the first row of the orbit
        orbit[0] = t0, *tuple(self.r0), *tuple(self.v0)
        # this above is equivalent to 
        # orbit[0] = t0, self.r0[0], self.r0[1], self.r0[2], self.v0[0], self.v0[1], self.v0[2]
        
        
        # initialize a counter for the orbit.  
        i = 1 # since we already set the 0th values, we start the counter at 1
        r, v = self.r0, self.v0
        
        # start the integration (advancing in time steps and computing LeapFrog at each step)
        while t < tmax:  # as long as t has not exceeded the maximal time 
            
            # **** advance the time by one timestep, dt
            t += dt
            # **** store the new time in the first column of the ith row
            orbit[i,0] = t
            
            # ***** advance the position and velocity using the LeapFrog scheme
            # remember that LeapFrog returns a position vector and a velocity vector  
            # as an example, if a function returns three vectors you would call the function and store 
            # the variable like:     a,b,c = function(input)
            r, v = self.LeapFrog(r,v,dt)
         
    
            # ****  store the new position vector into the columns with indexes 1,2,3 of the ith row of orbit
            # TIP:  if you want columns 5-7 of the Nth row of an array called A, you would write : 
            # A[n, 5:8] 
            # where the syntax is row n, start at column 5 and end BEFORE column 8
            orbit[i, 1:4] = r
            
            # ****  store the new position vector into the columns with indexes 1,2,3 of the ith row of orbit
            orbit[i, 4:7] = v
            
            # **** update counter i , where i is keeping track of the number of rows (i.e. the number of time steps)
            i += 1
        
        
        # write the data to a file
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#', 
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        
        # there is no return function
        

#Compute the Analytic Orbit
filename = "M33_analytic_orbit.txt"
t0 = 0      # Initial time (Gyr)
dt = 0.1    # timestep
tmax = 10   # Max time (Gyr)

# Create an instance and run integration
M33_orbit = M33AnalyticOrbit(filename)
M33_orbit.OrbitIntegration(t0, dt, tmax)

# Load the M33 analytic orbit
analytic_orbit = np.loadtxt(filename)

#Load in simulation data from HW6
M31_data = np.loadtxt("Orbit_M31.txt")  
M33_data = np.loadtxt("Orbit_M33.txt")  

#Compute Relative Orbit from Simulation
t_simulation = M31_data[:, 0]  # Time in Gyr
r_M31 = M31_data[:, 1:4]       # Position of M31
r_M33 = M33_data[:, 1:4]       # Position of M33
v_M31 = M31_data[:, 4:7]       # Velocity of M31
v_M33 = M33_data[:, 4:7]       # Velocity of M33

# Compute relative position and velocity
r_relative = np.linalg.norm(r_M33 - r_M31, axis=1)
v_relative = np.linalg.norm(v_M33 - v_M31, axis=1)

# Step 4: Extract Analytic Data
t_analytic = analytic_orbit[:, 0]
r_analytic = np.linalg.norm(analytic_orbit[:, 1:4], axis=1)
v_analytic = np.linalg.norm(analytic_orbit[:, 4:7], axis=1)

# Step 5: Plot Total Position Over Time
plt.figure(figsize=(10, 5))
plt.plot(t_analytic, r_analytic, label="Analytic M33 Orbit")
plt.plot(t_simulation, r_relative, label="Simulation M33 Orbit HW6", linestyle='dashed')
plt.xlabel("Time (Gyr)")
plt.ylabel("Total Position (kpc)")
plt.legend()
plt.title("M33 Orbit: Analytic vs Simulation")
plt.grid()
plt.show()

# Step 6: Plot Total Velocity Over Time
plt.figure(figsize=(10, 5))
plt.plot(t_analytic, v_analytic, label="Analytic M33 Velocity")
plt.plot(t_simulation, v_relative, label="Simulation M33 Velocity HW6", linestyle='dashed')
plt.xlabel("Time (Gyr)")
plt.ylabel("Total Velocity (km/s)")
plt.legend()
plt.title("M33 Velocity: Analytic vs Simulation")
plt.grid()
plt.show()

# Discussion Questions
print("2. How do the plots compare?")
print("The two graphs show an orbiting feature with the spring like motion\
      however the simulation HW6 gets tighter over time, meaning it's being pulled in to something.\
      whereas the Analytical plot just shows a very long, probably stable orbit.")

print("\n3. What missing physics could make the difference?")
print("In the simulation HW6 we had the Milky Way galaxy involved in the simulation, which is a huge\
      gravitational thing to lose out on with the Analytical plots which only looked at M33 with M31.\
      If we added in the MW to the analytical steps we\'d see a massive difference in plots.")

print("\n4. The MW is missing in these calculations. How might you include its effects?")
print("I could add an additional acceleration term in the M31Accel to include the MW gravity effects.")
