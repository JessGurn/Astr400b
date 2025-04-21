# The Morphology of the MW-M31 Merger Remnant: Ending Shape
# Jessica Gurney
# ASTR 400B Research Assignment 3 
# 4/3/2025
# Updated ReadFile, CenterofMass, OrbitCOM with small checks to work with this program

'''
Topic: Investigating the morphology of the MW-M31 merger remnant.
Question: Does the remnant classify as a classical elliptical galaxy (E0–E7), and how does its shape vary with radius?
This script will generate a 2D histogram of the remnants stellar density, and fitting ellipses to density contours, then
seeing how those density contours change with radius, and change the classification of the galaxy. 
'''

# ===========================================
# Step 1: Import necessary libraries
# ===========================================
import numpy as np
import matplotlib.pyplot as plt
import os
from CenterOfMass import CenterOfMass  # my iterative COM code
from ReadFile import Read  # to load snapshot data
from GalaxyMass import ComponentMass as COM # to generate mass of galaxy
from OrbitCOM import orbitCOM 
from astropy import units as u
from skimage.measure import EllipseModel
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter


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

def load_galaxy_data(filename, ptypes=(2,3)):
    """Loads the data from the file and filters out unwanted particles
    INPUTS:
    -------
    filename: 'str'
        Name of the data file
    ptypes : 'tuple' of int
        Particle types we want to include (disk = 2, bulge = 3)
               
    RETURNS:
    --------
    filtered: 'array'
        Particle data from selected types
    """
    
    time, total, data = Read(filename)
    # Filter only stellar particles (disk=2, bulge=3) from MW and M31
    filtered = data[(data['type'] == ptypes[0]) | (data['type'] == ptypes[1])]
    return filtered


def merge_stars(dataMW, dataM31):
    """ Combines MW and M31 filtered data into a single array
    
    INPUTS:
    -------
    dataMW: 'array'
        Filtered data from MW
    
    dataM31: 'array'
        Filtered data from M31

    RETURNS:
    --------
    merged : 'array'
        Combined array of all stellar particles
    """
    merged = np.concatenate((dataMW, dataM31), axis=0)
    #merged['m'] = np.abs(merged['m'])  # Fix negative masses
    return merged


def save_merged_file(filename, merged_stars, snapshot):
    """ Saves the merged galaxy into a snapshot format to be used for READFILE.pyy
    
    INPUTS:
    -------
    filename: 'str'
        Name of the file were creating ("merged_700.txt")
    
    merged: 'array'
        Combined stellar particles data
        
    snapshot: 'int'
        Snapshot number used in the header (e.g. 700)

    RETURNS:
    --------
    merged : 'array'
        Combined array of all stellar particles
    """
    # Save array with the correct headers and as plain data array
    data_to_save = np.column_stack((
        merged_stars['type'].astype(int),
        merged_stars['m'],
        merged_stars['x'], merged_stars['y'], merged_stars['z'],
        merged_stars['vx'], merged_stars['vy'], merged_stars['vz']
    ))
    
    # open the file and write in the headers
    with open(filename, "w") as f:
        f.write(f"Time {snapshot}.0\n")
        f.write(f"Total {len(merged_stars)}\n")
        f.write("mass in 1e10, x, y, z, in kpc and vx, vy, vz in km/s\n")
        f.write("type m x y z vx vy vz\n")
    
    # Append the merged data using "a" mode and defining number display 
    with open(filename, "a") as f:
        np.savetxt(f, data_to_save, fmt=["%d", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e", "%.6e"])
    

# ===========================================
# Step 4: Calculate the Center of Mass (COM)
# ===========================================
# Use CenterOfMass.py to find the iterative COM position

# Use COM class on merged stellar particles (type 2 and 3)
# Had to update Center of mass file to allow both
# Compute COM position using disk (type 2)


def mass_weighted_COM(filename):
    """ Center of mass will be based on position/velocity and mass of the combined
    disk and bulge particles.

    INPUTS:
    -------
    filename: 'str'
        Name of the merged snapshot file
    
    RETURNS:
    --------
    pos_COM : 'array'
        Mass-weighted center of mass position [kpc]
    
    vel_COM: 'array'
        Mass-weighted center of mass velocity [km/s]
    """
    # Disk Center of Mass
    com_disk = CenterOfMass(filename, 2)
    pos_disk = com_disk.COM_P(0.1)
    vel_disk = com_disk.COM_V(pos_disk[0], pos_disk[1], pos_disk[2])
    mass_disk = np.sum(com_disk.m)
    
    # Bulge Center of Mass
    com_bulge = CenterOfMass(filename, 3)
    pos_bulge = com_bulge.COM_P(0.1)
    vel_bulge = com_bulge.COM_V(pos_bulge[0], pos_bulge[1], pos_bulge[2])
    mass_bulge = np.sum(com_bulge.m)
    
    # Mass weighted position
    # Center depends not just where things are places but how heavy they are
    # Mtot = Mdisk + Mbulge
    # r_COM = (M_disk*r_disk + M_bulge*r_bulge)/(Mdisk + Mbulge)
    total_mass = mass_disk + mass_bulge
    x_COM = (pos_disk[0].value * mass_disk + pos_bulge[0].value * mass_bulge) / total_mass
    y_COM = (pos_disk[1].value * mass_disk + pos_bulge[1].value * mass_bulge) / total_mass
    z_COM = (pos_disk[2].value * mass_disk + pos_bulge[2].value * mass_bulge) / total_mass
    com_pos = np.array([x_COM, y_COM, z_COM]) * u.kpc

    # Mass weighted velocity
    vx_COM = (vel_disk[0].value * mass_disk + vel_bulge[0].value * mass_bulge) / total_mass
    vy_COM = (vel_disk[1].value * mass_disk + vel_bulge[1].value * mass_bulge) / total_mass
    vz_COM = (vel_disk[2].value * mass_disk + vel_bulge[2].value * mass_bulge) / total_mass
    com_vel = np.array([vx_COM, vy_COM, vz_COM]) * u.km/u.s
    
    return com_pos, com_vel
    
    

# ===========================================
# Step 5: Center and align angular momentum with z-axis
# ===========================================
# Align angular momentum vector along z-axis (to get face-on view)
# Right now angular momentum is pointing in any direction
# Figure out how far off it is from Z axis - rotate it to align to Z axis using a rotation Matrix
# Use np.cross and rotation matrix
# Compute the angular momentum vector L = sum(m * (r x v))

def shift_rotate(data, com_pos, com_vel):
    """ Places COM position and velocity at origin (0,0,0). Finds direction of 
    angular moment moment and aligns it with z axis. 

    INPUTS:
    -------
    data: 'array'
        Particle data from merged snapshot
        
    com_pas: 'array'
        COM position [kpc]
    
    com_vel: 'array'
        COM velocity [km/s]
        
    RETURNS:
    --------
    x, y, z, vx, vy, vz, m: 'arrays'
        Shifted and rotated particles
    """
    # Shift all positions and velocities so COM is at (0,0,0) in both position and velocity
    x = data['x'] - com_pos[0].value
    y = data['y'] - com_pos[1].value
    z = data['z'] - com_pos[2].value
    vx = data['vx'] - com_vel[0].value
    vy = data['vy'] - com_vel[1].value
    vz = data['vz'] - com_vel[2].value
    m = data['m']          # particle masses, shape (N,)
    
    # Create position and velocity vectors (already COM-centered from Step 4)
    r = np.array([x, y, z])        # shape (3, N)
    v = np.array([vx, vy, vz])     # shape (3, N)
    
    # Initialize angular momentum vector
    L = np.zeros(3)

    # Loop through particles to compute total L = sum(m_i*(r_i x v_i))
    for i in range(len(m)):
        ri = r[:, i]   # [x_i, y_i, z_i]
        vi = v[:, i]   # [vx_i, vy_i, vz_i]
        L += m[i] * np.cross(ri, vi)

    # Normalize L to get the direction of angular momentum
    L_mag = np.linalg.norm(L)

    if L_mag == 0:
        print("Warning: zero angular momentum vector.")
        L_norm = np.array([0, 0, 1])
    else:
        L_norm = L / L_mag

    # Define the unit vector i z direction
    z_hat = np.array([0, 0, 1])

    # Rotation axis and angle
    v_axis = np.cross(L_norm, z_hat)        #Rotation axis(finds angle perpendicular to both Z and L)
    sin_angle = np.linalg.norm(v_axis)      #Angle between angular momentum and z
    cos_angle = np.dot(L_norm, z_hat)       #Angle between angular momentum and z

    # If not already aligned, compute rotation matrix using Rodrigues' formula
    if sin_angle != 0:
        v_axis /= sin_angle  # normalize the rotation axis

        # Skew-symmetric cross-product matrix of v_axis
        K = np.array([
            [0, -v_axis[2], v_axis[1]],
            [v_axis[2], 0, -v_axis[0]],
            [-v_axis[1], v_axis[0], 0]
        ])

        # Rodrigues rotation formula: R = I + K + K^2 * ((1 - cosθ)/sin^2θ)
        I = np.eye(3)   #Identity matrix
        R = I + K + K @ K * ((1 - cos_angle) / (sin_angle**2))

        # Apply rotation to position and velocity
        r_rot = R @ r
        v_rot = R @ v

        # Get the new rotated components
        x, y, z = r_rot[0], r_rot[1], r_rot[2]
        vx, vy, vz = v_rot[0], v_rot[1], v_rot[2]

    else:
        print("No rotation needed; already aligned with z-axis.")

    # Sanity check: print final direction of L
    L_check = np.zeros(3)
    for i in range(len(m)):
        ri = np.array([x[i], y[i], z[i]])
        vi = np.array([vx[i], vy[i], vz[i]])
        L_check += m[i] * np.cross(ri, vi)

    L_check_norm = L_check / np.linalg.norm(L_check)
    
    print("Post-rotation L direction:", L_check_norm)
    
    return r[0], r[1], r[2], v[0], v[1], v[2], m
    



# ===========================================
# Step 6: Create a 2D histogram (density map) in the XY plane
# ===========================================
# - Use np.histogram2d or plt.hist2d to plot the surface density
# - Define bin edges and extent for visualization

def plot_projection(a, b, alabel, blabel, title):
    """ Plts a 2D histogram (density plot) Image only in a plane such as 
    face on, side, top-down to visualize the overall structure of the galaxy 
    if viewed from telescope and colorized based on density. 
    
    INPUTS:
    -------
    a: 'array'
        the "x" axis array on the generated plot in [kpc]
        
    b: 'array'
        the "y" axis array on the generated plot in [kpc]
        
    alabel: 'string'
        Title for "x" axis (known as a) to us in plot
        
    blabel: 'string'
        Title for "y" axis (known as b) to us in plot
        
    title: 'string'
        Title for the plot, recommended callout x vs y string
           
    RETURNS:
    --------  
    image: 'array'
        Created surface density histogram in log-scale
    
    xedges/yedges: 'array'
        Bin edges along the axis, xedge[0] is left edge, xedge[1] is right edge of bin
    """
    
    # Create blank figure, specify Histogram bins (500 x 500 grid)
    plt.figure(figsize=(8, 6))
    nbins = 500                 
    extent = [[-50, 50], [-50, 50]]  # X and Y range in kpc (-50 -> +50 kpc)

    # Create Histogram, x and y are centered and rotated from step 5
    # If using weight it includes total mass per pixel (mass density) instead of just star density
    # H is number of particles in each bin
    H, xedges, yedges = np.histogram2d(a, b, bins=nbins, range=extent, weights=None)
    image = np.log10(H.T + 1)

    # Plot with log scale color
    # np.log10(H.t transposes histogram so axes match, +1 avoids log(0))
    # extent sets the axies to real splace coordinates
    plt.imshow(np.log10(H.T + 1), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='inferno')
    plt.colorbar(label='log10(Surface Density)')
    plt.xlabel(f"{alabel} [kpc]")
    plt.ylabel(f"{blabel} [kpc]")
    plt.title(title)
    plt.grid(False)
    plt.tight_layout()
    plt.show()
    
    return image, xedges, yedges



# ===========================================
# Step 7: Fit ellipses to the 2D density map using photutils
# ===========================================
# Use sigma_clipped_stats to get image background/noise estimate
# Define EllipseGeometry and pass to Ellipse to sample isophotes
# Measure semi-major (a) and semi-minor (b) axes for each ellipse
# Calculate axis ratio: b/a
# From b/a, classify the E-type (E0 to E7)

def density_contour(adata, a_title, bdata, b_title, nbins, title):
    """ Creates confidence contours over 2D histogram. The drawn contours represent
    (40% of stars in this region) 
    
    INPUTS:
    -------
    adata: 'array'
        the "x" axis array on the generated plot in [kpc]
        
    bdata: 'array'
        the "y" axis array on the generated plot in [kpc]
        
    alabel: 'string'
        Title for "x" axis (known as a) to us in plot
        
    blabel: 'string'
        Title for "y" axis (known as b) to us in plot
        
    nbins: 'int'
        The number of bins for the histogram, aka resolution grid (50x50 example)
        
    title: 'string'
        Title for the plot, recommended callout x vs y string
           
    RETURNS:
    --------  
    contour: 'contour'
        All contour lines at each confidence level, x,y coordinates, 
        line segments, color and style setting
    """
    smooth_sigma = 2    #Strength of Gaussian smoothing of contour lines, low to high
    extent=[[-50, 50], [-50, 50]]   #Histogram/Plot axis limits in [kpc]
    contour_colors = ['black', 'purple', 'red', 'orange', 'yellow', 'white', 'black', 'purple']
    
    # H is density per bin
    # xedge/yeges is the bin boundaries 
    H, xedges, yedges = np.histogram2d(adata, bdata, bins=nbins, range=extent, density=True)
    
    # Change density per unit area into probabibility per bin (Probability Density Function)
    # Remove density units: PDFij = H[i,j] * deltaXi * deltayj
    x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1, nbins))
    y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins, 1))
    prob_df = (H * (x_bin_sizes * y_bin_sizes))


    # Gaussian smoothing and matrix transposing
    Z = prob_df.T
    Z_smooth = gaussian_filter(Z, sigma=smooth_sigma)
    
    X = 0.5 * (xedges[1:] + xedges[:-1])
    Y = 0.5 * (yedges[1:] + yedges[:-1])

    # Compute contour levels for confidence levels
    flat = Z_smooth.flatten() #flatten 2d array to 1d for simplified sorting
    flat /= np.sum(flat)  # Normalize the prob_df to 1
    sorted_vals = np.sort(flat)[::-1]  #Sort flattened values highest -> Lowest to find most prob region
    cumsum = np.cumsum(sorted_vals) #Each pixel in order, how much total prob has been added up to that point
    
    thresholds = [0.10, 0.25, 0.45, 0.65, 0.80, 0.90, 0.95, 0.99]   #confidence levels we want to visualze
    # finds index where cumsum first equals or exceeds threshold
    # value returned is min prob_df height so sum of all pixels above that includes at least t of the prob
    levels = [sorted_vals[np.where(cumsum >= t)[0][0]] for t in thresholds][::-1] 
    strs = ['0.10', '0.25', '0.45', '0.65', '0.80', '0.90', '0.95', '0.99'][::-1]
    fmt = {}
    
    #fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,6), sharex=True, sharey=True)
    from matplotlib import gridspec

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.05, 0.25, 1], wspace=0.05)
    
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[3])
    cax = fig.add_subplot(gs[1])  # for the colorbar

    # Create histogram
    # left panel
    hist = ax1.hist2d(adata, bdata, bins=500, range=extent, norm=LogNorm(), cmap='inferno')
    #hist = ax.hist2d(adata, bdata, bins=500, range=extent, norm=LogNorm(), cmap='inferno')
    cbar = plt.colorbar(hist[3], cax=cax)
    cbar.set_label("log10(Surface Density)")

    ax1.set_title(title)
    ax1.set_xlabel(f"{a_title} [kpc]")
    ax1.set_ylabel(f"{b_title} [kpc]")
    ax1.set_xlim(extent[0])
    ax1.set_ylim(extent[1])

    # Draw the contours enclosing the levels of probability specified
    contour = ax1.contour(X, Y, Z_smooth, levels=levels, origin='lower', colors=contour_colors)
    # Annotate contour lines
    for l, s in zip(contour.levels, strs):
        fmt[l] = s
    ax1.clabel(contour, contour.levels, inline=True, fmt=fmt, fontsize=10)
    
    #Right panel ellipses and overlay
    contour2 = ax2.contour(X, Y, Z_smooth, levels=levels, origin='lower', colors="gray")
    ax2.set_title(f"Fitted Ellipses over Contours \n{title} Plan")
    ax2.set_xlabel(f"{a_title} [kpc]")
    ax2.set_ylabel(f"{b_title} [kpc]")
    ax2.legend(["Contour"], loc="upper right", fontsize=8)
    ax2.set_aspect("equal")
    ax2.set_xlim(extent[0])
    ax2.set_ylim(extent[1])

    # Fit ellipses and overlay
    ellipse_data, ellipse_lines = ellipticity(contour2) #call function for ellipses
    #plot_ba_vs_radius_bar(ellipse_data, a_title, b_title)  #Plot the E# Data
    
    print(f"Drawing {len(ellipse_lines)} ellipses")  # Debug output
    for x_ell, y_ell in ellipse_lines:
        ax2.plot(x_ell, y_ell, '--', linewidth=2)

    plt.tight_layout()

    plt.show()

    return ellipse_data



# ===========================================
# Step 9: Plot axis ratio as a function of radius
# ===========================================
# Show how galaxy shape changes with radius
# Useful to detect triaxiality or elongation in the outskirts
# Save axis ratio profile to file
# POSSIBLE TO GET THE CONTOURS THAT WERE PLOTTED, THEIR SHAPE, AND DO THE 
# AXIS RATIO THING
# tHIS IS THE FUNCTION I WRITE MYSELF
# Use the contours, the contour lines are the a/b 
# Can generate plot with radius's on x, value of a/b on y for ellipticity 
# E types are spherical, so need to also compare the sphericality between planes. 
# Normalize longest plane to 1, and get percentage of other two planes. 
# Do pie chart for 3 axis. Perfect thirds perfect spherical, (chart per contour)
# Stacked vertical bar with normalization to 1. So 3 would be perfect sphere. 



#Works plots by itself

def ellipticity(contours):
    """ Fit one ellipse per contour level by combining all paths at that level''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''' """

   
    ellipse_data = []
    ellipse_lines = []  #for drawing plot in density contour

    for i, collection in enumerate(contours.collections):
        print(f"\n--- Contour Level {i} (PDF = {contours.levels[i]:.4e}) ---")
        paths = collection.get_paths()

        # Combine all vertices from all paths at this level
        all_vertices = []
        for path in paths:
            all_vertices.append(path.vertices)

        if len(all_vertices) == 0:
            print("  No valid paths at this level.")
            continue

        combined = np.vstack(all_vertices)  # Stack all x,y points into one array

        model = EllipseModel()
        
        if model.estimate(combined):
            xc, yc, a, b, theta = model.params
            t = np.linspace(0, 2 * np.pi, 300)
            ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
            ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
            
            
            a, b = max(a, b), min(a, b)
            ba_ratio = b / a
            e_number = int(round(10 * (1 - ba_ratio)))
            r = np.sqrt(ellipse_x**2 + ellipse_y**2) #distance from center
            r_mean = float(np.mean(r))
            
            ellipse_data.append({"a":a, "b":b, "E":e_number, "b/a":ba_ratio, "r":r_mean})
            ellipse_lines.append((ellipse_x, ellipse_y))
            
            print(f"  Combined Fit: b/a = {ba_ratio:.3f}, E{e_number}")
        else:
            print("  Combined ellipse fit failed.")
            
    
    return ellipse_data, ellipse_lines

# ===========================================
# Step 10: Summarize findings
# ===========================================
# Print or save axis ratio values and E0-E7 classification at different radii
# Interpret whether the remnant is spheroidal, elliptical, or triaxial

def plot_ba_vs_radius_bar(xy, xy_label, xz, xz_label, yz, yz_label):
    """Create bar plot of b/a vs average radius.'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''"""
    
   
    xy_data = sorted(xy, key=lambda x: x["r"])
    xz_data = sorted(xz, key=lambda x: x["r"])
    yz_data = sorted(yz, key=lambda x: x["r"])

    
    radii = [entry["r"] for entry in xy_data]
    xy_ratios = [entry["b/a"] for entry in xy_data]
    xz_ratios = [entry["b/a"] for entry in xz_data]
    yz_ratios = [entry["b/a"] for entry in yz_data]
    
    xy_E = [entry["E"] for entry in xy_data]
    xz_E = [entry["E"] for entry in xz_data]
    yz_E = [entry["E"] for entry in yz_data]

    # Grouped bar spacing
    x = np.arange(len(radii))
    bar_width = 0.25

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - bar_width, xy_ratios, width=bar_width, label=xy_label, color='Gold')
    bars2 = plt.bar(x,           xz_ratios, width=bar_width, label=xz_label, color='LightCoral')
    bars3 = plt.bar(x + bar_width, yz_ratios, width=bar_width, label=yz_label, color='DarkMagenta')

    # Optional E-type labels on top of each bar
    for bar, E in zip(bars1, xy_E):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.015, f"E{E}", ha='center', va='bottom', fontsize=8)
    for bar, E in zip(bars2, xz_E):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.015, f"E{E}", ha='center', va='bottom', fontsize=8)
    for bar, E in zip(bars3, yz_E):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.015, f"E{E}", ha='center', va='bottom', fontsize=8)

    # Format axes
    plt.xticks(x, [f"{r:.1f}" for r in radii], rotation=45)
    plt.xlabel("Average Radius [kpc]")
    plt.ylabel("Axis Ratio (b/a)")
    plt.title("Axis Ratio vs Radius\nGrouped by Projection Plane")
    plt.grid(True, axis='y')
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    plt.show()



def filter_core_particles(x, y, z, vx, vy, vz, m, max_radius=50):
    """
    Filters particles to include only those within a specified radius of the origin (COM).
    
    INPUTS:
    -------
    x, y, z : arrays of positions [kpc]
    vx, vy, vz : arrays of velocities [km/s]
    m : array of particle masses
    max_radius : float
        Maximum distance from origin to include [kpc]
    
    RETURNS:
    --------
    x, y, z, vx, vy, vz, m : arrays of filtered particles
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    idx = np.where(r < max_radius)
    return x[idx], y[idx], z[idx], vx[idx], vy[idx], vz[idx], m[idx]



def main():
    snapshot = 600
    
    mw_file = f"MW/MW_{snapshot}.txt"
    m31_file = f"M31/M31_{snapshot}.txt"
    merged_file = f"merged_{snapshot}.txt"
    
    dataMW = load_galaxy_data(mw_file)
    dataM31 = load_galaxy_data(m31_file)
    merged_data = merge_stars(dataMW, dataM31)
    save_merged_file(merged_file, merged_data, snapshot)

    com_pos, com_vel = mass_weighted_COM(merged_file)

    # Shift + rotate
    x, y, z, vx, vy, vz, m = shift_rotate(merged_data, com_pos, com_vel)

    #Galaxy only images, unhide 
    #xy_image, xface, yface = plot_projection(x, y, alabel="x", blabel="y", title="Face On XY Projection \n MW-M31 Merger Remnant")
    #xz_image, xtop, ytop = plot_projection(x, z, alabel="x", blabel="z", title="Top Down XZ Projection \n MW-M31 Merger Remnant")
    #yz_image, xside, yside = plot_projection(y, z, alabel="y", blabel="z", title="Side View YZ Projection \n MW-M31 Merger Remnant")
    
    # Optional: remove outer particles > max_radius to clean up plots
    x, y, z, vx, vy, vz, m = filter_core_particles(x, y, z, vx, vy, vz, m, max_radius=50)
    
    
    xy_contour = density_contour(x, "x", y, "y", 80, "Face on XY \n Contour Plot Projection")
    xz_contour = density_contour(x, "x", z, "z", 80, "Top Down XZ \n Contour Plot Projection")
    yz_contour = density_contour(y, "y", z, "z", 80, "Side View on YZ \n Contour Plot Projection")
    
    plot_ba_vs_radius_bar(xy_contour, "XY Plane (Face on)", xz_contour, "XZ Plane (Top Down)", yz_contour, "YZ Plane (Side View)")  #Plot the E# Data
    

    

if __name__ == "__main__":
    main()