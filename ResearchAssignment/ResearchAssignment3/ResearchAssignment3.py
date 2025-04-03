# The Morphology of the MW-M31 Merger Remnant: Ending Shape
# Jessica Gurney
# ASTR 400B Research Assignment 3 
# 4/3/2025
# Updated ReadFile, CenterofMass2, OrbitCOM with small checks to work with this program

'''
Topic: Investigating the morphology of the MW-M31 merger remnant.
Question: Does the remnant classify as a classical elliptical galaxy (E0–E7), and how does its shape vary with radius?
This script will generate a 2D histogram of the remnants stellar density, and fitting ellipses to quantify the axis ratio.
'''

# ===========================================
# Step 1: Import necessary libraries
# ===========================================
import numpy as np
import matplotlib.pyplot as plt
import os
from CenterOfMass2 import CenterOfMass  # my iterative COM code
from ReadFile import Read  # to load snapshot data
from GalaxyMass import ComponentMass as COM # to generate mass of galaxy
from OrbitCOM import orbitCOM # updated from HW solution to work for Research code

# from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
# from photutils.isophote import EllipseSample
# from photutils.aperture import EllipticalAperture
# from astropy.io import ascii
# from astropy.stats import sigma_clipped_stats



# ===========================================
# Step 2: If files aren't found run the MW-M31 merger simulations
# ===========================================

# Only run GalaxyMass and OrbitCOM if orbit files aren't found.
# Need large snapshot library of MW and M31 

if not os.path.exists("Orbit_MW.txt") or not os.path.exists("Orbit_M31.txt"):
    
    # Run Galaxymass.py for both MW and M31 to get disk type ==2 and bulge mass type ==3 for total mass
    M31_disk = COM('M31_000.txt', 2)
    M31_bulge = COM('M31_000.txt', 3)
    
    MW_disk = COM('MW_000.txt', 2)
    MW_bulge = COM('MW_000.txt', 3)
    
    # Run OrbitCOM.py for MW and M31 separately to get position/velocity over time/orbit files
    # Saves the orbit file to directory
    orbitCOM(galaxy='MW', start=0, end=800, n=5) 
    orbitCOM(galaxy='M31', start=0, end=800, n=5)
    
    # Create a plot using generated orbit files to identify the merger time (when seperation is near zero)
    MW_data = np.genfromtxt("Orbit_MW.txt", comments="#")
    M31_data = np.genfromtxt("Orbit_M31.txt", comments="#")

    # Compute relative separation
    # Column 0 is time in Gyr
    # Column 1-3 = x,y,z positon of COM
    # Colun 4-6 = vx, vy, vz
    # MW_data[:, 1:4] gives an array of shape (snapshots, 3) of position vector for each step
    # seperation gives vectory difference of each time step then adds up square differenc for r^2 distance at each snapshot
    separation = np.sqrt(np.sum((MW_data[:,1:4] - M31_data[:,1:4])**2, axis=1))

    plt.figure()
    plt.plot(MW_data[:, 0], separation, label='MW–M31 Separation')
    plt.xlabel('Time [Gyr]')
    plt.ylabel('Separation [kpc]')
    plt.title('MW–M31 Separation Over Time')
    plt.grid(True)
    plt.legend()
    plt.show()
    
else:
    print("Orbit files found - skipping simulation rerun.")

# Choose one snapshot after the galaxies have merged manually by reading the graph
# Chose 7 Gyr snapshot 700 because it's post merger and settled. 




# ===========================================
# Step 3: Combine MW and M31 into one array
# ===========================================
# Use ReadFile.py to load the data for the seperate galaxies

time1, total1, dataMW = Read("MW/MW_700.txt")
time2, total2, dataM31 = Read("M31/M31_700.txt")

# Filter only stellar particles (disk=2, bulge=3) from MW and M31
MW_stars = dataMW[(dataMW['type'] == 2) | (dataMW['type'] == 3)] 
M31_stars = dataM31[(dataM31['type'] == 2) | (dataM31['type'] == 3)]

# Concatenate arrays for MW and M31 stellar components
merged_stars = np.concatenate((MW_stars, M31_stars), axis=0)

# Save merged array as new file with correct headers. 
# Prepare a plain data array to match the snapshot format
data_to_save = np.column_stack((
    merged_stars['x'], merged_stars['y'], merged_stars['z'],
    merged_stars['vx'], merged_stars['vy'], merged_stars['vz'],
    merged_stars['m'], merged_stars['type'].astype(int)
))

# manually write the 3 header lines expected by ReadFile.py
with open("merged_700.txt", "w") as f:
    f.write("t 700.0\n")
    f.write(f"N {len(merged_stars)}\n")
    f.write("x y z vx vy vz m type\n")

# Append the actual data using np.savetxt
with open("merged_700.txt", "a") as f:
    np.savetxt(f, data_to_save, fmt=["%.6e"]*7 + ["%d"])




# ===========================================
# Step 4: Calculate the Center of Mass (COM)
# ===========================================
# Use CenterOfMass2.py to find the iterative COM position

# Use COM class on merged stellar particles (type 2 and 3)
# Had to update Center of mass file to allow both
com_finder = CenterOfMass("merged_700.txt", [2,3])  
com_pos = com_finder.COM_P(0.1, 2)

print("COM position:", com_pos)


# Shift all positions so COM is at (0,0,0)
x = merged_stars['x'] - com_pos[0].value
y = merged_stars['y'] - com_pos[1].value
z = merged_stars['z'] - com_pos[2].value


print("Total particles in merged:", len(merged_stars))
print("Type counts:", np.unique(merged_stars['type'], return_counts=True))


# ===========================================
# Step 5: Center and align angular momentum with z-axis
# ===========================================
# Align angular momentum vector along z-axis (to get face-on view)
# Use np.cross and rotation matrix
# Compute the angular momentum vector L = sum(m * (r x v))

m = merged_stars['m']    #particle mass
vx = merged_stars['vx']  #velocity components
vy = merged_stars['vy']
vz = merged_stars['vz']

# Remove the bulk velocity, centers the velocity like centering position
# So I'm measuring angular momentum about center of mass frame
vx -= np.mean(vx)
vy -= np.mean(vy)
vz -= np.mean(vz)

# Create the Position and velocity vectors
r = np.array([x, y, z])  # position shape (3, N)
v = np.array([vx, vy, vz])  # velocity shape (3, N)

# Compute the angular momentum vector
# Cross product for each particle between r and v
# L = np.sum(m * np.cross(r.T, v.T), axis=0) (arrays didn't line up so do it manually)
# L = sum(m * (r x v))

L = np.zeros(3)  # Initialize total angular momentum vector

for i in range(len(m)):     #Loop through each particle to get momentum
    ri = np.array([x[i], y[i], z[i]])
    vi = np.array([vx[i], vy[i], vz[i]])
    L += m[i] * np.cross(ri, vi)
'''    
print("L before normalization:", L)
print("Norm of L:", np.linalg.norm(L))

L_norm = L / np.linalg.norm(L)   #Unit vector pointing direction of angular momentum
print("L_norm =", L_norm)
'''
# Normalize L and align with z-axis
L_mag = np.linalg.norm(L)

if L_mag == 0:
    print("Warning: zero angular momentum.")
    L_norm = np.array([0, 0, 1])
else:
    L_norm = L / L_mag
    
    
# Define rotation matrix to align L with z-axis, give the angle and axis between L and z
# Define unit z-axis
z_hat = np.array([0, 0, 1])    

v_axis = np.cross(L_norm, z_hat)  #Rotate the axis to be z
sin_angle = np.linalg.norm(v_axis)  #find sin between L and z
cos_angle = np.dot(L_norm, z_hat)   #find cos between L and z

# Construct the rotation matrix 
# Used Rodrigues 3x3 rotation formula:
# R = I + K + K^2(1-costheta/sin^2theta)
# R = rotation matrix
# K = skew-symmetric matrix k=v_axis   
# K = [0, -kz, ky
#      kz, 0, -kx
#     -ky, kx, 0]

if sin_angle != 0:  #0 would mean its already aligned
    v_axis /= sin_angle  #normalize rotation axis
    
    I = np.eye(3) #built in identity matrix
    
    K = np.array([[0, -v_axis[2], v_axis[1]],
                  [v_axis[2], 0, -v_axis[0]],
                  [-v_axis[1], v_axis[0], 0]])
    
    R = I + K + K @ K * ((1 - cos_angle) / (sin_angle ** 2))  #@ matrix multiplication
    
    rotated = R @ r     #apply rotation matrix to position
    
    x, y, z = rotated[0], rotated[1], rotated[2]    #define the x, y, and z of the rotated position
else:
    print("No rotation needed; angular momentum already aligned with z-axis.")
    
    
# Sanity check after rotation
r_rot = np.array([x, y, z])
v_rot = np.array([vx, vy, vz])
L_check = np.zeros(3)

for i in range(len(m)):
    ri = np.array([x[i], y[i], z[i]])
    vi = np.array([vx[i], vy[i], vz[i]])
    L_check += m[i] * np.cross(ri, vi)

L_check_norm = L_check / np.linalg.norm(L_check)
print("Post-rotation L_norm:", L_check_norm)


# ===========================================
# Step 6: Create a 2D histogram (density map) in the XY plane
# ===========================================
# - Use np.histogram2d or plt.hist2d to plot the surface density
# - Define bin edges and extent for visualization

plt.figure(figsize=(8, 6))

# Define 2D histogram bins
# Total square will be 100x100 kpc across, divided by 300x300 bins ~0.33 kpc

nbins = 300
extent = [[-50, 50], [-50, 50]]  # X and Y range in kpc

# Create Histogram, x and y are centered and rotated from step 5
H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=extent)

# Plot with log scale color
# np.log10(H.t transposes histogram so axes match, +1 avoids log(0))

print(f"x range: {x.min():.2f} to {x.max():.2f}")
print(f"y range: {y.min():.2f} to {y.max():.2f}")


plt.imshow(np.log10(H.T + 1), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno')
plt.colorbar(label='log10(Surface Density)')
plt.xlabel('x [kpc]')
plt.ylabel('y [kpc]')
plt.title('Face-On Stellar Density of MW-M31 Merger Remnant')
plt.grid(False)
plt.tight_layout()
plt.show()

# YZ (side view)
plt.figure()
plt.scatter(y, z, s=0.5, alpha=0.5)
plt.xlabel("y [kpc]")
plt.ylabel("z [kpc]")
plt.title("Side View (YZ plane)")
plt.grid(True)
plt.show()

# XZ
plt.figure()
plt.scatter(x, z, s=0.5, alpha=0.5)
plt.xlabel("x [kpc]")
plt.ylabel("z [kpc]")
plt.title("Side View (XZ plane)")
plt.grid(True)
plt.show()



# ===========================================
# Step 7: Fit ellipses to the 2D density map using photutils
# ===========================================
# Use sigma_clipped_stats to get image background/noise estimate
# Define EllipseGeometry and pass to Ellipse to sample isophotes
# Measure semi-major (a) and semi-minor (b) axes for each ellipse
# Calculate axis ratio: b/a
# From b/a, classify the E-type (E0 to E7)


# ===========================================
# Step 8: Plot the density map with ellipse overlays
# ===========================================
# Overlay fitted ellipses on top of the 2D histogram
# Optionally, annotate the plot with ellipse parameters



# ===========================================
# Step 9: Plot axis ratio as a function of radius
# ===========================================
# Show how galaxy shape changes with radius
# Useful to detect triaxiality or elongation in the outskirts
# Save axis ratio profile to file


# ===========================================
# Step 10: Summarize findings
# ===========================================
# Print or save axis ratio values and E0-E7 classification at different radii
# Interpret whether the remnant is spheroidal, elliptical, or triaxial

