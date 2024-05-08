# Import packages.
import numpy as np
import pandas as pd
import scipy.spatial.distance as dist  # for computing pairwise distances
import utils


# Read in file with x,y,z and radius.
xyzr = pd.read_csv('jourdain_df1.78_kf1.5.xyz', sep='\t', skiprows=2, header=None)

xyzr = xyzr.to_numpy()  # convert to numpy array
xyzr = xyzr[xyzr[:, 2].argsort()]  # optional sort by z-axis


# Extract centers and radii.
c = xyzr[:, [0,1,2]]  # center of monomers
r = xyzr[:, [3]]  # radii

# Optionally scale to keep everything compact.
rm = np.mean(r)  # average radii
c = c / r
r = r / rm

#== OPTIONS ============================================================#
# Force parameters.
opts = dict()
opts['k'] = 0.2             # spring constant to center (larger pulls monomers more = more compact)
opts['k_decay'] = 1.000     # decay in spring constant (allows for freezing before full FCC structure)
opts['eps'] = 0.5           # well depth for Lennard-Jones (LJ), approx. Van der Waals force
opts['v_decay'] = 0.98      # decay of velocities at each timestep

opts['jit'] = 0.0           # controls jitter (0 = no jitter)
opts['jit_decay'] = 0.98    # decay in jitter

# Timestep and iterations.
opts['Dt'] = 0.01           # timestep
opts['nt'] = 600            # number of timesteps per block
opts['nj'] = 8              # number of blocks

# opts = utils.load_config('config.yml')  # alternatively, use and load config files (easier for saving conditions)
#=======================================================================#



# Initialize quantities relevant to the simulation.
# Second LJ parameter. Get from radii.
sig = (r + np.transpose(r)) / (2 ** (1/6))

# Initialize velocity as zeros.
v = np.array([0, 0, 0])

# Equivalent mass.
m = (4/3*np.pi) * r ** 3;

# Initialize positions.
x = c;

# Copy over options that act as variables tha change over iterations or recur.
jit = opts['jit']
k = opts['k']
Dt = opts['Dt']

# Loop through a simulation to collapse the aggregates.
for jj in range(opts['nj']):
    for ii in range(opts['nt']):
        jit = jit * opts['jit_decay']  # incorporate decay of forces
        k = k * opts['k_decay']
    
        # Initialize force.
        cen = utils.com(x, r)  # recompute center of mass
        F = -opts['k'] * (x - cen)     # force toward center of the aggregate
        
        d0 = dist.squareform(dist.pdist(x))
        d0[d0 == 0] = np.Inf  # avoids division by zero below
        
        d = x[..., np.newaxis]  # add third dimension

        d = np.transpose(d, (0, 2, 1)) - np.transpose(d, (2, 0, 1))  # pairwise distances (used for force direction)

        ds = np.sqrt(np.sum(d ** 2, axis=2))[..., np.newaxis]  # to normalize distances
        ds[ds == 0] = np.Inf  # avoids division by zero

        d = d / ds  # normalize to get unit direction
        
        # Force is derivative of the Lennard-Jones potential.
        Fvw = 48 * opts['eps'] * ((sig / d0) ** 12 / d0 - 0.5 * (sig / d0) ** 6 / d0)
        Fvw[d0 > 1.5 * sig] = 0  # cutoff of 1.5 * sig (only closest monomers)
        
        # Sum forces between aggregates.
        F = F + np.sum(Fvw[..., np.newaxis] * d, axis=1)
        
        # Add random perturbation to avoid getting stuck.
        F = F + jit * np.random.randn(*F.shape)

        # Stepping.        
        # Compute acceleration.
        a = F / m
        
        # Leap frog integration.
        v = v + a * Dt  # compute new velocities based on acceleration
        x = x + v * Dt  # compute new positions based on velocity

        v = v * opts['v_decay'];   # dampen velocity

utils.write_xyz(x, r, 'collapsed.xyz')  # write result to xyz file

utils.render(x, r)  # render an image of the collapsed aggregate
